#include "gkh.h"
#include "givens.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <pthread.h>
#include <stdexcept>
#include <thread>
#include <vector>

namespace
{
    static Matrix transpose_copy(const Matrix &A)
    {
        Matrix T(A.cols(),A.rows(),0.0);
        for (int i=0;i<A.rows();++i){
            for (int j=0;j<A.cols();++j)
            {
                T.at(j,i)=A.at(i,j);
            }
        }
        return T;
    }

    // 活动块 [l, r]（闭区间）表示一个尚未完全收敛的上二对角子问题。
    // 在该区间内，超对角线元素非零，你可以认为通过这个抽象结构给矩阵“分块”。
    struct Block
    {
        int l;
        int r;
    };

    // 对矩阵 M 的两行 r0, r1 左乘 Givens 旋转 [c s; -s c]。
    // 即 M <- L * M，其中 L 只作用在第 r0/r1 两行上。
    // 这类逐元素线性组合很适合向量化，SIMD/多线程中你也可以顺手的事把他们做了。
    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        int col=M.cols();
        int j=0;
        double *row0=&M.at(r0,j);
        double *row1=&M.at(r1,j);
        float64x2_t vc=vdupq_n_f64(c);
        float64x2_t vs=vdupq_n_f64(s);
        float64x2_t vns=vdupq_n_f64(-s);
        //>八路
        for (; j+7<col;j+=8)
        {
            float64x2_t a0=vld1q_f64(row0+j);//load
            float64x2_t a1=vld1q_f64(row0+j+2);
            float64x2_t a2=vld1q_f64(row0+j+4);
            float64x2_t a3=vld1q_f64(row0+j+6);
            float64x2_t b0=vld1q_f64(row1+j);
            float64x2_t b1=vld1q_f64(row1+j+2);
            float64x2_t b2=vld1q_f64(row1+j+4);
            float64x2_t b3=vld1q_f64(row1+j+6);
            float64x2_t new00=vfmaq_f64(vmulq_f64(vs,b0),vc,a0);//multi and sum
            float64x2_t new01=vfmaq_f64(vmulq_f64(vc,b0),vns,a0);
            float64x2_t new10=vfmaq_f64(vmulq_f64(vs,b1),vc,a1);
            float64x2_t new11=vfmaq_f64(vmulq_f64(vc,b1),vns,a1);
            float64x2_t new20=vfmaq_f64(vmulq_f64(vs,b2),vc,a2);
            float64x2_t new21=vfmaq_f64(vmulq_f64(vc,b2),vns,a2);
            float64x2_t new30=vfmaq_f64(vmulq_f64(vs,b3),vc,a3);
            float64x2_t new31=vfmaq_f64(vmulq_f64(vc,b3),vns,a3);
            vst1q_f64(row0+j,new00);//store back
            vst1q_f64(row1+j,new01);
            vst1q_f64(row0+j+2,new10);
            vst1q_f64(row1+j+2,new11);
            vst1q_f64(row0+j+4,new20);
            vst1q_f64(row1+j+4,new21);
            vst1q_f64(row0+j+6,new30);
            vst1q_f64(row1+j+6,new31);
        }
        //>四路
        for (; j+3<col;j+=4)
        {
            float64x2_t a0=vld1q_f64(row0+j);//load
            float64x2_t a1=vld1q_f64(row0+j+2);
            float64x2_t b0=vld1q_f64(row1+j);
            float64x2_t b1=vld1q_f64(row1+j+2);
            float64x2_t new00=vfmaq_f64(vmulq_f64(vs,b0),vc,a0);//multi and sum
            float64x2_t new01=vfmaq_f64(vmulq_f64(vc,b0),vns,a0);
            float64x2_t new10=vfmaq_f64(vmulq_f64(vs,b1),vc,a1);
            float64x2_t new11=vfmaq_f64(vmulq_f64(vc,b1),vns,a1);
            vst1q_f64(row0+j,new00);//store back
            vst1q_f64(row1+j,new01);
            vst1q_f64(row0+j+2,new10);
            vst1q_f64(row1+j+2,new11);
        }
        //>二路
        for(;j+1<col;j+=2){
            float64x2_t a=vld1q_f64(row0+j);
            float64x2_t b=vld1q_f64(row1+j);
            float64x2_t new0=vfmaq_f64(vmulq_f64(vs,b),vc,a);
            float64x2_t new1=vfmaq_f64(vmulq_f64(vc,b),vns,a);
            vst1q_f64(row0+j,new0);
            vst1q_f64(row1+j,new1);
        }
        //>一路
        for(;j<col;j++){
            double a = M.at(r0, j);
            double b = M.at(r1, j);
            M.at(r0, j) = c * a + s * b;
            M.at(r1, j) = -s * a + c * b;
        }
    }

    // 对矩阵 M 的两列 c0, c1 右乘 Givens 旋转 [c s; -s c]。
    // 即 M <- M * R，其中 R 只作用在第 c0/c1 两列上。
    static void apply_right_cols_scalar(Matrix &M, int c0, int c1, double c, double s)
    {
        for (int i=0;i<M.rows();++i)
        {
            double a=M.at(i,c0);
            double b=M.at(i,c1);
            M.at(i,c0)=a*c-b*s;
            M.at(i,c1)=a*s+b*c;
        }
    }

    //对转置矩阵M^T的两行执行左乘，等价于原矩阵M的右乘两列。
    static void apply_right_cols(Matrix &Mt, int c0, int c1, double c, double s)
    {
        apply_left_rows(Mt,c0,c1,c,-s);
    }

    static void accumulate_left_into_U(Matrix &Ut, int r0, int r1, double c, double s)
    {
        // 我们该怎样积累 U 和 V 的更新呢？
        // 以此处 U 的积累为例，让我们B <- L * B 时，我们必须维护的等式是 A = U * B * V^T
        // 如果 A = U * B * V^T 不成立，那么我们最终的SVD结果显然不是 A 的正确分解。
        // 由于正交矩阵和其转置的乘积是I，一个自然的想法是让 U <- U * L^T。
        // 这样就变成 A = (U * L^T) * (L * B) * V^T = U * B * V^T，等式得以保持。

        // U <- U * L^T  <=>  U^T <- L * U^T，直接复用行连续访问的 SIMD 路径。
        apply_left_rows(Ut,r0,r1,c,s);
    }

    // 计算活动块 [l, r] 对应 B^T B 右下 2x2 主子块的 Wilkinson 偏移。
    // 偏移用于加速 QR 迭代收敛，并让 bulge chasing 过程更稳定。
    static double block_wilkinson_shift(const Matrix &B, int l, int r)
    {
        if (r == l)
        {
            return B.at(l, l) * B.at(l, l);
        }

        const double d1 = B.at(r - 1, r - 1);
        const double e1 = B.at(r - 1, r);
        const double d2 = B.at(r, r);
        const double e0 = (r - 1 > l) ? B.at(r - 2, r - 1) : 0.0;

        const double a = d1 * d1 + e0 * e0;
        const double b = d1 * e1;
        const double d = d2 * d2 + e1 * e1;

        const double tr = a + d;
        const double det = a * d - b * b;
        double disc = 0.25 * tr * tr - det;
        if (disc < 0.0)
        {
            disc = 0.0;
        }

        const double root = std::sqrt(disc);
        const double lam1 = 0.5 * tr + root;
        const double lam2 = 0.5 * tr - root;
        return (std::fabs(lam1 - d) <= std::fabs(lam2 - d)) ? lam1 : lam2;
    }

    // 将上二对角结构以外、且绝对值很小的元素强制置零。
    static void cleanup_bidiagonal(Matrix &B, double tol)
    {
        for (int i = 0; i < B.rows(); ++i)
        {
            for (int j = 0; j < B.cols(); ++j)
            {
                if (j != i && j != i + 1 && std::fabs(B.at(i, j)) <= tol)
                {
                    B.at(i, j) = 0.0;
                }
            }
        }
    }

    // 对活动块 [l, r] 执行一次“单块 GKH bulge chasing”迭代。
    // 流程：首次右乘引入 bulge -> 首次左乘消 bulge -> 交替右乘/左乘将 bulge 追赶到块末端。
    static void one_block_step(Matrix &Ut, Matrix &B, Matrix &Vt, int l, int r)
    {
        //block 中不可能存在 r<l
        //r=l 1*1 不需要左右乘
        if (r <= l)
        {
            return;
        }
        //加速的 miu
        const double mu=block_wilkinson_shift(B,l,r);

        double c = 1.0;
        double s = 0.0;
        double rr = 0.0;

        // 首次右乘：由 (d_l^2-mu, d_l*e_l) 构造。
        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);
        //计算 givens 矩阵 false 是右乘
        givens_rotation(x,z,c,s,rr,false);
        apply_right_cols_scalar(B,l,l+1,c,s);
        //对 V 累乘
        apply_right_cols(Vt,l,l+1,c,s);

        // 首次左乘：消去 (l+1, l)。
        givens_rotation(B.at(l,l),B.at(l+1,l),c,s,rr,true);
        apply_left_rows(B,l,l+1,c,s);
        //对 U 累乘
        accumulate_left_into_U(Ut,l,l+1,c,s);

        for (int k = l + 1; k <= r - 1; ++k)
        {
            // 右乘：消去 (k-1, k+1)
            givens_rotation(B.at(k-1,k),B.at(k-1,k+1),c,s,rr,false);
            apply_right_cols_scalar(B,k,k+1,c,s);
            apply_right_cols(Vt,k,k+1,c,s);

            // 左乘：消去 (k+1, k)
            givens_rotation(B.at(k,k),B.at(k+1,k),c,s,rr,true);
            apply_left_rows(B,k,k+1,c,s);
            accumulate_left_into_U(Ut,k,k+1,c,s);
        }
    }

    // 处理“对角元 d_k 近零但超对角 e_k 未近零”的情况。
    // 思路与单块追赶类似：先右乘把 e_i 消掉，再左乘清理新引入的次对角 bulge，
    // 把这个问题逐步向右传递，直到块末端。
    static bool chase_zero_diagonal(Matrix &Ut, Matrix &B, Matrix &Vt, int k, double tol)
    {
        const int m = B.rows();
        const int n = B.cols();
        if (k < 0 || k >= n - 1)
        {
            return false;
        }

        // d_k ~ 0 且 e_k 还未收敛时，按 lim_1 思路进行压缩追赶：
        // 1) 右乘消去第 k 行的 e_k；2) 左乘消去引入的次对角 bulge；
        // 然后把问题传递到下一行，直到末端。
        if (std::fabs(B.at(k, k + 1)) <= tol)
        {
            return false;
        }

        bool changed = false;
        for (int i = k; i <= n - 2; ++i)
        {
            double c = 1.0;
            double s = 0.0;
            double rr = 0.0;

            // 右乘：使第 i 行满足 [d_i, e_i] * G = [r, 0]。
            givens_rotation(B.at(i,i),B.at(i,i+1),c,s,rr,false);
            apply_right_cols_scalar(B,i,i+1,c,s);
            apply_right_cols(Vt,i,i+1,c,s);

            // 左乘：消去 (i+1, i) 处由右乘引入的 bulge。
            if (i + 1 < m)
            {
                givens_rotation(B.at(i,i),B.at(i+1,i),c,s,rr,true);
                apply_left_rows(B,i,i+1,c,s);
                accumulate_left_into_U(Ut,i,i+1,c,s);
            }

            changed = true;
        }

        cleanup_bidiagonal(B, tol);
        return changed;
    }

    // 扫描所有 d_k≈0 的位置：若对应 e_k 仍显著非零，则调用追赶过程压缩该异常结构。
    // 返回值表示本轮是否对 B/U/V 做了实际更新。
    static bool handle_diagonal_zeros(Matrix &Ut, Matrix &B, Matrix &Vt, double tol)
    {
        const int n = B.cols();
        bool changed = false;

        const double eps = std::numeric_limits<double>::epsilon();
        const double diag_tol = tol;
        const double super_tol = tol * (1.0 + 10.0 * eps);

        for (int k = 0; k < n - 1; ++k)
        {
            if (std::fabs(B.at(k, k)) <= diag_tol && std::fabs(B.at(k, k + 1)) > super_tol)
            {
                if (chase_zero_diagonal(Ut,B,Vt,k,tol))
                {
                    changed = true;
                }
            }
        }

        return changed;
    }

    // 根据超对角线是否“足够小”对问题进行分块。
    // 若 |e_k| <= tol*(|d_k|+|d_{k+1}|+1)，认为该位置可解耦并直接置零。
    // 最终会得到一系列小矩阵。
    static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol)
    {
        for (int k = 0; k < n - 1; ++k)
        {
            const double a = std::fabs(B.at(k, k));
            const double d = std::fabs(B.at(k + 1, k + 1));
            const double crit = tol * (a + d + 1.0);
            if (std::fabs(B.at(k, k + 1)) <= crit)
            {
                B.at(k, k + 1) = 0.0;
            }
        }

        std::vector<Block> blocks;
        int l = 0;
        while (l < n)
        {
            //一开始都是 0，0
            int r = l;
            //除非执行到最后 或者 上次对角线上的数为 0就分块
            while (r < n - 1 && std::fabs(B.at(r, r + 1)) > 0.0)
            {
                ++r;
            }
            //返回说明在 l,l)(r,r)左右闭合的里面是一个子矩阵
            blocks.push_back({l, r});
            l = r + 1;
        }
        return blocks;
    }

    // 收尾步骤：
    // 1) 把奇异值（对角元）统一调整为非负；
    // 2) 按降序重排奇异值，同时同步重排 U、V 对应列。
    // 最终得到常见的 SVD 规范形式：sigma_1 >= sigma_2 >= ... >= 0。
    // 这个函数你不用太在意，后续任务也不会明确涉及它。
    static void make_nonnegative_and_sort(Matrix &U, Matrix &B, Matrix &V)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < n; ++i)
        {
            if (B.at(i, i) < 0.0)
            {
                B.at(i, i) = -B.at(i, i);
                for (int r = 0; r < m; ++r)
                {
                    U.at(r, i) = -U.at(r, i);
                }
            }
        }

        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i)
        {
            idx[i] = i;
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return B.at(a, a) > B.at(b, b); });

        Matrix U2 = U;
        Matrix V2 = V;
        Matrix D(B.rows(), B.cols(), 0.0);

        for (int new_i = 0; new_i < n; ++new_i)
        {
            const int old_i = idx[new_i];
            D.at(new_i, new_i) = B.at(old_i, old_i);

            for (int r = 0; r < U.rows(); ++r)
            {
                U2.at(r, new_i) = U.at(r, old_i);
            }
            for (int r = 0; r < V.rows(); ++r)
            {
                V2.at(r, new_i) = V.at(r, old_i);
            }
        }

        U = U2;
        V = V2;
        B = D;
    }

    // 在子区间[l,r]内继续切块。
    static std::vector<Block> split_subblocks_in_range(Matrix &B,int l,int r,double tol)
    {
        for (int k=l;k<r;++k)
        {
            const double a=std::fabs(B.at(k,k));
            const double d=std::fabs(B.at(k+1,k+1));
            const double crit=tol*(a+d+1.0);
            if (std::fabs(B.at(k,k+1))<=crit)
            {
                B.at(k,k+1)=0.0;
            }
        }
        std::vector<Block>out;
        int cur=l;
        while (cur<=r)
        {
            int rr=cur;
            while (rr<r&&std::fabs(B.at(rr,rr+1))>0.0)
            {
                ++rr;
            }
            out.push_back({cur,rr});
            cur=rr+1;
        }
        return out;
    }

    static int detect_worker_count(int task_count)
    {
        const unsigned hc=std::thread::hardware_concurrency();
        const int cores=(hc==0U)?4:static_cast<int>(hc);
        if (task_count<=0)
        {
            return 1;
        }
        return std::max(1,std::min(cores,task_count));
    }

    //线程数初始化
    static int gkh_manual_threads()
    {
        constexpr int manual_threads=2;
        return manual_threads;
    }

    static int choose_init_threads(int expected_parallel)
    {
        const int auto_threads=detect_worker_count(expected_parallel);
        const int manual=gkh_manual_threads();
        if (manual>0)
        {
            return manual;
        }
        return auto_threads;
    }

    struct GkhThreadPool;
    //每一个工作线程只拿一个指向对应 struct 的参数就行
    struct WorkerArg
    {
        GkhThreadPool *pool;
    };

    //定义 GKH 线程池
    //相当于总控台
    struct GkhThreadPool
    {
        int thread_count=0;
        //工人集合
        std::vector<pthread_t>workers;
        //作为参数传入线程函数 void * XXX(void*args)
        std::vector<WorkerArg>args;
        //🔒
        pthread_mutex_t mu=PTHREAD_MUTEX_INITIALIZER;
        //唤醒使用
        pthread_cond_t cv_job=PTHREAD_COND_INITIALIZER;
        //结束使用
        pthread_cond_t cv_done=PTHREAD_COND_INITIALIZER;
        //线程池是否终止
        bool stop=false;
        //是否触发全局步数上限
        bool limit_hit=false;
        //是否完成当前批次
        bool done=false;
        //当前活跃工人数
        int active_workers=0;
        //全局步数统计与上限（一次 one_block_step 记为 1 步）
        long long total_steps=0;
        long long max_global_steps=0;
        //单任务本地步数上限
        int max_local_steps=0;
        //收敛阈值
        double tol=1e-12;
        //工人用的数据
        Matrix *Ut=nullptr;
        Matrix *B=nullptr;
        Matrix *Vt=nullptr;
        //异步任务队列
        std::deque<Block>task_queue;
    };

    // 线程拿到一个子块后，持续做 gkh；若分裂出新子块则回投队列。
    static void process_block_async(GkhThreadPool *pool,const Block &start_blk)
    {
        Block cur=start_blk;
        int local_steps=0;
        while (cur.r>cur.l)
        {
            one_block_step(*pool->Ut,*pool->B,*pool->Vt,cur.l,cur.r);
            local_steps++;
            //更新 pool（一次上锁同时更新步数与上限状态，减少锁开销）
            bool hit_limit=false;
            pthread_mutex_lock(&pool->mu);
            pool->total_steps++;
            if (pool->max_global_steps>0&&pool->total_steps>=pool->max_global_steps)
            {
                pool->limit_hit=true;
                hit_limit=true;
            }
            pthread_mutex_unlock(&pool->mu);
            //防止一直算
            if (hit_limit)
            {
                return;
            }
            // 单任务连续推进过长时回投，避免单块长期占用一个线程。
            if (pool->max_local_steps>0&&local_steps>=pool->max_local_steps)
            {
                pthread_mutex_lock(&pool->mu);
                pool->task_queue.push_back(cur);
                pthread_cond_broadcast(&pool->cv_job);
                pthread_mutex_unlock(&pool->mu);
                return;
            }
            // 子块内部再次扫描：若出现可切分断点，立即拆成更小任务。
            std::vector<Block>parts=split_subblocks_in_range(*pool->B,cur.l,cur.r,pool->tol);
            Block keep={-1,-1};
            int keep_len=-1;
            for (const auto &blk:parts)
            {
                if (blk.r>blk.l)
                {
                    const int len=blk.r-blk.l+1;
                    if (len>keep_len)
                    {
                        keep_len=len;
                        keep=blk;
                    }
                }
            }
            // 其余非平凡块回投队列，让空闲线程来认领。
            int pushed=0;
            pthread_mutex_lock(&pool->mu);
            for (const auto &blk:parts)
            {
                if (blk.r<=blk.l)
                {
                    continue;
                }
                if (blk.l==keep.l&&blk.r==keep.r)
                {
                    continue;
                }
                pool->task_queue.push_back(blk);
                ++pushed;
            }
            if (pushed>0)
            {
                pthread_cond_broadcast(&pool->cv_job);
            }
            pthread_mutex_unlock(&pool->mu);
            if (keep.r<=keep.l)
            {
                return;
            }
            cur=keep;
        }
    }

    //调度工人执行
    static void *gkh_worker_loop(void *raw)
    {
        WorkerArg *arg=static_cast<WorkerArg*>(raw);
        GkhThreadPool *pool=arg->pool;
        while (true)
        {
            pthread_mutex_lock(&pool->mu);
            //睡，等着被叫
            while (pool->task_queue.empty()&&!pool->stop&&!pool->limit_hit)
            {
                if (pool->active_workers==0)
                {
                    pool->done=true;
                    pthread_cond_signal(&pool->cv_done);
                }
                pthread_cond_wait(&pool->cv_job,&pool->mu);
            }
            if (pool->stop||pool->limit_hit)
            {
                //停下就收工解锁
                pthread_mutex_unlock(&pool->mu);
                break;
            }

            //get var
            Block blk=pool->task_queue.front();
            pool->task_queue.pop_front();
            ++pool->active_workers;
            //保证拿的数据都 ok 好的解锁计算
            pthread_mutex_unlock(&pool->mu);
            //分块天然的优势
            process_block_async(pool,blk);
            //上锁更新
            pthread_mutex_lock(&pool->mu);
            --pool->active_workers;
            if (pool->limit_hit)
            {
                pool->done=true;
                pthread_cond_signal(&pool->cv_done);
                pthread_cond_broadcast(&pool->cv_job);
            }
            else if (pool->task_queue.empty()&&pool->active_workers==0)
            {
                pool->done=true;
                pthread_cond_signal(&pool->cv_done);
            }
            pthread_mutex_unlock(&pool->mu);
        }
        return nullptr;
    }
    //初始化
    static bool init_pool(GkhThreadPool &pool,int threads)
    {
        //单线程别并行了，串行就好
        if (threads<=1)
        {
            return false;
        }
        pool.thread_count=threads;
        pool.workers.resize(threads);
        pool.args.resize(threads);

        for (int i=0;i<threads;i++)
        {
            pool.args[i].pool=&pool;
            //worker地址 null 执行的函数 传的参数地址
            //如果失败
            if (pthread_create(&pool.workers[i],nullptr,gkh_worker_loop,&pool.args[i])!=0)
            {
                //创建失败，线程结束
                pool.stop=true;
                //全叫醒，并且清除
                pthread_cond_broadcast(&pool.cv_job);
                for (int j=0;j<i;j++)
                {
                    pthread_join(pool.workers[j],nullptr);
                }
                return false;
            }
        }
        return true;
    }

    static void grow_pool_best_effort(GkhThreadPool &pool,int target_threads)
    {
        if (target_threads<=pool.thread_count)
        {
            return;
        }
        const int old_count=pool.thread_count;
        pool.workers.resize(target_threads);
        pool.args.resize(target_threads);
        int created=0;
        for (int i=old_count;i<target_threads;++i)
        {
            pool.args[i].pool=&pool;
            if (pthread_create(&pool.workers[i],nullptr,gkh_worker_loop,&pool.args[i])!=0)
            {
                break;
            }
            ++created;
        }
        pool.thread_count=old_count+created;
        pool.workers.resize(pool.thread_count);
        pool.args.resize(pool.thread_count);
    }

    static void destroy_pool(GkhThreadPool &pool)
    {
        //单线程没开不用的destroy
        if (pool.thread_count<=1)
        {
            return;
        }
        //上锁防更改
        pthread_mutex_lock(&pool.mu);
        pool.stop=true;
        pthread_cond_broadcast(&pool.cv_job);
        pthread_mutex_unlock(&pool.mu);
        //合并
        for (int i=0;i<pool.thread_count;i++)
        {
            pthread_join(pool.workers[i], nullptr);
        }
    }

    static void run_blocks_with_pool(GkhThreadPool &pool,Matrix &Ut,Matrix &B,Matrix &Vt,const std::vector<Block> &jobs)
    {
        //blocks为空就不用执行
        if (jobs.empty())
        {
            return;
        }
        //线性池初始化
        pthread_mutex_lock(&pool.mu);
        if (pool.limit_hit)
        {
            pthread_mutex_unlock(&pool.mu);
            return;
        }
        pool.Ut=&Ut;
        pool.B=&B;
        pool.Vt=&Vt;
        for (const auto &blk:jobs)
        {
            if (blk.r>blk.l)
            {
                pool.task_queue.push_back(blk);
            }
        }
        pool.done=false;
        pthread_cond_broadcast(&pool.cv_job);
        while (!pool.done&&!pool.limit_hit)
        {
            pthread_cond_wait(&pool.cv_done,&pool.mu);
        }
        pthread_mutex_unlock(&pool.mu);
    }

    static bool pool_limit_reached(GkhThreadPool &pool)
    {
        pthread_mutex_lock(&pool.mu);
        bool hit=pool.limit_hit;
        pthread_mutex_unlock(&pool.mu);
        return hit;
    }

} // namespace


// 从“上二对角矩阵 B”出发执行 Golub-Kahan SVD 迭代（改进版）：
// - 输入输出满足 A = U * B * V^T 不变；
// - 迭代中自动分块、处理对角近零、并在每个活动块上做 bulge chasing；
// - 成功收敛后，B 被整理为非负且降序的对角矩阵（其对角元即奇异值）。
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: requires m >= n");
    }
    if (U.rows() != m || U.cols() != m)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: U must be m x m");
    }
    if (V.rows() != n || V.cols() != n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: V must be n x n");
    }

    bool converged=false;
    Matrix Ut=transpose_copy(U);
    Matrix Vt=transpose_copy(V);

    //防卡死上限：按“总步数”而非“全局迭代轮数”限制
    const long long global_step_cap=static_cast<long long>(max_iter);
    GkhThreadPool pool;
    bool pool_inited=false;
    bool pool_ok=false;
    long long serial_total_steps=0;
    
    while (true)
    {
        // 清理数值噪声，并优先处理 d_k≈0 的特殊情形。
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(Ut,B,Vt,tol);

        // 根据超对角线断点拆分活动块
        // 这里子矩阵间是相互独立的，所以此处具有很大的并行潜力：你可以尝试多线程/多进程进行处理
        // 但根据算法，收集 Givens 旋转并更新 U/V 需要在每个块内顺序执行，所以这可能给并行带来麻烦。
        std::vector<Block> blocks = split_active_blocks(B, n, tol);

        //统计非平凡块数量
        int active_cnt=0;
        for (const auto&blk:blocks)
        {
            if (blk.r>blk.l)
            {
                ++active_cnt;
            }
        }

        //没有非平凡块：已收敛
        if (active_cnt==0)
        {
            converged=true;
            break;
        }
        // 非平凡块长度至少为 2，因此理论并行块上限约为 n/2。
        const int max_possible_parallel=std::max(1,n/2);
        // 按当前分块数量的 2 倍估计需求，给后续分裂留一点余量。
        const int expected_parallel=std::max(1,std::min(max_possible_parallel,active_cnt*2));
        const int desired_threads=choose_init_threads(expected_parallel);

        if (!pool_inited)
        {
            pool_ok=init_pool(pool, desired_threads);
            pool_inited=pool_ok;
            if (pool_ok)
            {
                pthread_mutex_lock(&pool.mu);
                pool.max_global_steps=global_step_cap;
                // 更积极地回投单块任务，给其它线程更多“偷到子块”的机会。
                pool.max_local_steps=std::max(32,n/16);
                pool.tol=tol;
                pthread_mutex_unlock(&pool.mu);
            }
        }
        else if (pool_ok && desired_threads>pool.thread_count)
        {
            grow_pool_best_effort(pool,desired_threads);
        }
        int size=static_cast<int>(blocks.size());
        //没创建好线程就串行
        if (!pool_ok)
        {
            for (int i=size-1;i>=0;--i)
            {
                if (blocks[i].r>blocks[i].l)
                {
                    one_block_step(Ut,B,Vt,blocks[i].l,blocks[i].r);
                    ++serial_total_steps;
                    if (serial_total_steps>=global_step_cap)
                    {
                        break;
                    }
                }
            }
            if (serial_total_steps>=global_step_cap)
            {
                break;
            }
            continue;
        }
        //创建 block
        std::vector<Block>jobs;
        jobs.reserve(active_cnt);
        for (int i=size-1;i>=0;i--)
        {
            if (blocks[i].r>blocks[i].l)
            {
                jobs.push_back(blocks[i]);
            }
        }
        //多线程
        run_blocks_with_pool(pool,Ut,B,Vt,jobs);
        if (pool_limit_reached(pool))
        {
            break;
        }
    }
    destroy_pool(pool);

    U=transpose_copy(Ut);
    V=transpose_copy(Vt);

    // 迭代结束后统一结构清理与标准化输出。
    cleanup_bidiagonal(B, tol);
    for (int i = 0; i < n - 1; ++i)
    {
        B.at(i, i + 1) = 0.0;
    }
    make_nonnegative_and_sort(U, B, V);

    return converged;
}
