from numba import njit,float64,int32
import numpy as np
STEP=1e5

@njit(cache=True)
def _forward_push_mean(indptr,indices,degree,s,eps,alpha, debug=False):
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int32)
    front, rear = np.int32(0), np.int32(2)
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = s
    queue[2] = n
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    q_mark[n] = 1
    eps_vec = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = np.float64(0.)
    sum_r = np.float64(1.)
    active_nodes = 0
    epoch_num = 0
    freq_com = np.zeros(n, dtype=np.float64)

    l1_error = []
    num_oper_list = []

    mean_r = -np.infty
    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:
            mean_r = 0.
            for i in range(rear - front):
                vv = queue[((front + 1) % n) + i]
                mean_r += (r[vv] / degree[vv])
            mean_r /= (rear - front)
            mean_r -= 1e-15
            l1_error.append(sum_r)
            num_oper_list.append(num_oper)
            epoch_num += 1
            rear = (rear + 1) % n
            queue[rear] = n
            active_nodes = 0
            continue
        if (r[u] / degree[u]) < mean_r:
            rear = (rear + 1) % n
            queue[rear] = u
            q_mark[u] = True
            continue
        freq_com[u] += 1
        num_oper += degree[u]
        res_u = r[u]
        sum_r -= alpha * res_u
        gpr_vec[u] += alpha * res_u
        rest_prob = (1. - alpha) / degree[u]
        r[u] = 0
        push_amount = rest_prob * res_u
        for v in indices[indptr[u]:indptr[u + 1]]:
            r[v] += push_amount
            if not q_mark[v] and np.abs(r[v]) >= eps_vec[v]:
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True
                active_nodes += 1

    l1_error.append(sum_r)
    num_oper_list.append(num_oper)
    epoch_num=len(l1_error)

    re = np.zeros(n + 2*epoch_num +1, dtype=np.float64)
    re[:n] = gpr_vec
    re[n:n+epoch_num] = l1_error
    re[n+epoch_num:-1] = num_oper_list
    re[-1]= epoch_num
    return re

def forward_push_mean(adj_matrix,s,eps,alpha):
    degree=np.int32(adj_matrix.sum(1).A.flatten())
    indices=adj_matrix.indices
    indptr=adj_matrix.indptr
    eps=eps/indptr[-1]
    vec_num_op=_forward_push_mean(indptr,indices,degree,s,eps,1-alpha)
    
    n=adj_matrix.shape[0]
    epoch_num=int(vec_num_op[-1])
    gpr_vec=vec_num_op[:n]
    l1_error=vec_num_op[n:n+epoch_num]
    num_oper_list=vec_num_op[n+epoch_num:-1]
    return gpr_vec,num_oper_list,l1_error


@njit(cache=False)
def _forward_push(indptr,indices,degree,s,eps,alpha, debug=False):
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int64)
    front, rear = np.int64(0), np.int64(2)
    gpr_vec, r = np.zeros(n, dtype=np.float64), np.zeros(n + 1, dtype=np.float64)
    queue[1] = s
    queue[2] = n
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    q_mark[n] = 1
    eps_vec = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        eps_vec[i] = degree[i] * eps

    num_oper = np.float64(0.)
    sum_r = np.float64(1.)
    active_nodes = 0
    epoch_num = 0

    l1_error = []
    num_oper_list = []
    step=1e4
    threshold = 1e4

    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:
            l1_error.append(sum_r)
            num_oper_list.append(num_oper)
            epoch_num += 1
            rear = (rear + 1) % n
            queue[rear] = n
            active_nodes = 0
            continue
        active_nodes += 1
        num_oper += degree[u]            
        res_u = r[u]
        sum_r -= alpha * res_u

        if num_oper>threshold:
            l1_error.append(sum_r)
            num_oper_list.append(num_oper)
            threshold += step


        gpr_vec[u] += alpha * res_u
        rest_prob = (1. - alpha) / degree[u]
        r[u] = 0
        push_amount = rest_prob * res_u
        for v in indices[indptr[u]:indptr[u + 1]]:
            r[v] += push_amount
            if not q_mark[v] and np.abs(r[v]) >= eps_vec[v]:
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True

    l1_error.append(sum_r)
    num_oper_list.append(num_oper)
    epoch_num=len(l1_error)

    re = np.zeros(n + 2*epoch_num +1, dtype=np.float64)
    re[:n] = gpr_vec
    re[n:n+epoch_num] = l1_error
    re[n+epoch_num:-1] = num_oper_list
    re[-1]= epoch_num
    return re


def forward_push(adj_matrix,s,eps,alpha):
    degree=np.int64(adj_matrix.sum(1).A.flatten())
    indices=adj_matrix.indices
    indptr=adj_matrix.indptr
    eps=eps/indptr[-1]
    vec_num_op=_forward_push(indptr,indices,degree,s,eps,1-alpha)

    n=adj_matrix.shape[0]
    epoch_num=int(vec_num_op[-1])
    gpr_vec=vec_num_op[:n]
    l1_error=vec_num_op[n:n+epoch_num]
    num_oper_list=vec_num_op[n+epoch_num:-1]
    return gpr_vec,num_oper_list,l1_error

@njit(float64[:](int32[:],int32[:],int32[:],int32,float64,float64),cache=True)
def _power_iteration(indptr,indices,degree,s,eps,alpha):
    n=degree.shape[0]
    m=indptr[-1]
    gpr_vec=np.zeros(n)
    gpr_vec[s]=1
    
    p_vec=np.zeros(n)
    p_vec[s]=1
    count=np.int32(0)
    num_oper=0.

    l1_error = []
    num_oper_list = []
    step=STEP
    threshold = step

    while count<2000:
        mat_x_vec=np.zeros(n)
        for u in np.arange(n):
            prob_mount=alpha*gpr_vec[u]/degree[u]
            for index in indices[indptr[u]:indptr[u+1]]:
                mat_x_vec[index]+=prob_mount
                
        sum=np.float64(0)
        for u in np.arange(n):
            sum+=mat_x_vec[u]
        l1_norm=np.float64(0)
        for u in np.arange(n):
            new_val=mat_x_vec[u]+(1-alpha)*p_vec[u]
            l1_norm+=np.abs(new_val-gpr_vec[u])
            gpr_vec[u]=new_val
        num_oper+=m

        if num_oper>threshold:
            l1_error.append(l1_norm)
            num_oper_list.append(num_oper)
            threshold += step

        if l1_norm<eps:
            break
        count+=1

    l1_error.append(l1_norm)
    num_oper_list.append(num_oper)
    epoch_num=len(l1_error)

    re = np.zeros(n + 2*epoch_num +1, dtype=np.float64)
    re[:n] = gpr_vec
    re[n:n+epoch_num] = l1_error
    re[n+epoch_num:-1] = num_oper_list
    re[-1]= epoch_num
    return re

def power_iteration(adj_matrix,s,eps,alpha):
    degree=np.int32(adj_matrix.sum(1).A.flatten())
    indices=adj_matrix.indices
    indptr=adj_matrix.indptr
    vec_num_op=_power_iteration(indptr,indices,degree,s,eps,alpha)

    n=adj_matrix.shape[0]
    epoch_num=int(vec_num_op[-1])
    gpr_vec=vec_num_op[:n]
    l1_error=vec_num_op[n:n+epoch_num]
    num_oper_list=vec_num_op[n+epoch_num:-1]
    return gpr_vec,num_oper_list,l1_error

@njit(float64[:](int32[:],int32[:],int32[:],int32,float64,float64),cache=True)
def _heavy_ball(indptr,indices,degree,s,eps,alpha):
    n=degree.shape[0]
    m=indptr[-1]
    
    kappa=np.sqrt((1-alpha)/(1+alpha))
    kappa_pow=np.power(1+kappa,2)
    eta=2*(1+np.power(kappa,2))/kappa_pow
    c1=alpha*eta
    c2=np.power(1-kappa,2)/kappa_pow
    c3=(1-alpha)*eta
    
    p_vec=np.zeros(n)
    p_vec[s]=1
    
    init_vec=p_vec.copy()
    x_pre=init_vec
    xt=init_vec
    nodes=np.arange(n)
    
    count=0    
    num_oper=0.

    l1_error = []
    num_oper_list = []
    step=STEP
    threshold = step

    while count<2000:
        gpr_vec=np.zeros(n)
        for u in nodes:
            prob_mount=xt[u]/degree[u]
            for index in indices[indptr[u]:indptr[u+1]]:
                gpr_vec[index]+=prob_mount
        for u in nodes:
            gpr_vec[u]=c1*gpr_vec[u]-c2*x_pre[u]+c3*p_vec[u]
        ell1_norm=0
        for u in nodes:
            ell1_norm+=np.abs(xt[u]-gpr_vec[u])
        num_oper+=m
        if ell1_norm<eps:
            break

        if num_oper>threshold:
            l1_error.append(ell1_norm)
            num_oper_list.append(num_oper)
            threshold += step

        x_pre=xt
        xt=gpr_vec
        count+=1

    l1_error.append(ell1_norm)
    num_oper_list.append(num_oper)
    epoch_num=len(l1_error)

    re = np.zeros(n + 2*epoch_num +1, dtype=np.float64)
    re[:n] = gpr_vec
    re[n:n+epoch_num] = l1_error
    re[n+epoch_num:-1] = num_oper_list
    re[-1]= epoch_num
    return re
    
def heavy_ball(adj_matrix,s,eps,alpha):
    degree=np.int32(adj_matrix.sum(1).A.flatten())
    indices=adj_matrix.indices
    indptr=adj_matrix.indptr
    vec_num_op=_heavy_ball(indptr,indices,degree,s,eps,alpha)

    n=adj_matrix.shape[0]
    epoch_num=int(vec_num_op[-1])
    gpr_vec=vec_num_op[:n]
    l1_error=vec_num_op[n:n+epoch_num]
    num_oper_list=vec_num_op[n+epoch_num:-1]
    return gpr_vec,num_oper_list,l1_error
    
@njit(float64[:](int32[:],int32[:],int32[:],int32,float64,float64),cache=True)
def _nag(indptr,indices,degree,s,eps,alpha):
    n=degree.shape[0]
    m=indptr[-1]
    
    eta=2/(2+alpha)
    c1=np.sqrt(4+2*alpha)
    c2=2*np.sqrt(1-alpha)
    beta=(c1-c2)/(c1+c2)
    c1=1-eta
    c2=alpha*eta
    
    c3=eta*(1-alpha)
    init_vec=np.zeros(n)
    init_vec[s]=1
    x_pre=init_vec
    xt=init_vec
    p_vec=init_vec.copy()
    zt=np.zeros(n)
    
    nodes=np.arange(n)
    count=0.
    num_oper=0.

    l1_error = []
    num_oper_list = []
    step=STEP
    threshold = step

    while count<2000:
        for u in nodes:
            zt[u]=xt[u]+beta*(xt[u]-x_pre[u])
        gpr_vec=np.zeros(n)
        for u in nodes:
            prob_mount=zt[u]/degree[u]
            for index in indices[indptr[u]:indptr[u+1]]:
                gpr_vec[index]+=prob_mount
        for u in nodes:
            gpr_vec[u]=c1*zt[u]+c2*gpr_vec[u]+c3*p_vec[u]
        l1_norm=0
        for u in nodes:
            l1_norm+=np.abs(xt[u]-gpr_vec[u])
        num_oper+=m
        if l1_norm<eps:
            break

        if num_oper>threshold:
            l1_error.append(l1_norm)
            num_oper_list.append(num_oper)
            threshold += step

        x_pre=xt
        xt=gpr_vec
        count+=1

    l1_error.append(l1_norm)
    num_oper_list.append(num_oper)
    epoch_num=len(l1_error)

    re = np.zeros(n + 2*epoch_num +1, dtype=np.float64)
    re[:n] = gpr_vec
    re[n:n+epoch_num] = l1_error
    re[n+epoch_num:-1] = num_oper_list
    re[-1]= epoch_num
    return re
    
def nag(adj_matrix,s,eps,alpha):
    degree=np.int32(adj_matrix.sum(1).A.flatten())
    indices=adj_matrix.indices
    indptr=adj_matrix.indptr
    vec_num_op=_nag(indptr,indices,degree,s,eps,alpha)

    n=adj_matrix.shape[0]
    epoch_num=int(vec_num_op[-1])
    gpr_vec=vec_num_op[:n]
    l1_error=vec_num_op[n:n+epoch_num]
    num_oper_list=vec_num_op[n+epoch_num:-1]
    return gpr_vec,num_oper_list,l1_error

@njit(float64[:](int32[:],int32[:],int32[:],int32,float64,float64),cache=True)
def _power_push(indptr,indices,degree,s,eps,alpha):    
    n=degree.shape[0]
    m=indptr[-1]
    queue=np.zeros(n,dtype=np.int32)
    front,rear=np.int32(0),np.int32(1)
    
    gpr_vec,res=np.zeros(n),np.zeros(n)
    queue[rear]=s    
    res[s]=1
    q_mark=np.zeros(n)
    q_mark[s]=1
    switch_size=np.int32(n/4)
    r_max=eps/m
    r_sum=1    
    eps_vec=r_max*degree
         
    num_oper=0.
    l1_error = []
    num_oper_list = []
    step=1e4
    threshold = step

    while front!=rear and ((rear-front)<=switch_size):        
        front=(front+1)%n
        u=queue[front]
        q_mark[u]=False
        if np.abs(res[u])>eps_vec[u]:
            alpha_residual=(1-alpha)*res[u]
            gpr_vec[u]+=alpha_residual
            r_sum-=alpha_residual
            increment=(res[u]-alpha_residual)/degree[u]
            res[u]=0
            num_oper+=degree[u]

            if num_oper>threshold:
                l1_error.append(r_sum)
                num_oper_list.append(num_oper)
                threshold += step

            for v in indices[indptr[u]:indptr[u+1]]:
                res[v]+=increment
                if not q_mark[v]:
                    rear=(rear+1)%n
                    queue[rear]=v
                    q_mark[v]=True
    jump=False
    if r_sum<=eps:
        jump=True
    num_epoch=8
      
    if not jump:
        for epoch in np.arange(1,num_epoch+1):
            r_max_prime1=np.power(eps,epoch/num_epoch)
            r_max_prime2=np.power(eps,epoch/num_epoch)/m
            while r_sum>r_max_prime1:
                u=0
                while u<n:
                    if res[u]>r_max_prime2*degree[u]:
                        alpha_residual=(1-alpha)*res[u]
                        gpr_vec[u]+=alpha_residual
                        r_sum-=alpha_residual
                        increment=(res[u]-alpha_residual)/degree[u]
                        res[u]=0
                        num_oper+=degree[u]

                        if num_oper>threshold:
                            l1_error.append(r_sum)
                            num_oper_list.append(num_oper)
                            threshold += step

                        for index in indices[indptr[u]:indptr[u+1]]:
                            res[index]+=increment
                    u+=1

    l1_error.append(r_sum)
    num_oper_list.append(num_oper)
    epoch_num=len(l1_error)

    re = np.zeros(n + 2*epoch_num +1, dtype=np.float64)
    re[:n] = gpr_vec
    re[n:n+epoch_num] = l1_error
    re[n+epoch_num:-1] = num_oper_list
    re[-1]= epoch_num
    return re

def power_push(adj_matrix,s,eps,alpha):
    degree=np.int32(adj_matrix.sum(1).A.flatten())
    indices=adj_matrix.indices
    indptr=adj_matrix.indptr
    vec_num_op=_power_push(indptr,indices,degree,s,eps,alpha)

    n=adj_matrix.shape[0]
    epoch_num=int(vec_num_op[-1])
    gpr_vec=vec_num_op[:n]
    l1_error=vec_num_op[n:n+epoch_num]
    num_oper_list=vec_num_op[n+epoch_num:-1]
    return gpr_vec,num_oper_list,l1_error
    
@njit(cache=True)
def _forward_push_sor(indptr,indices,degree,s,eps,alpha,omega):
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int64)
    front, rear = np.int64(0), np.int64(2)
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = s
    queue[2] = n  
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    q_mark[n] = 1
    eps_vec = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = np.float64(0.)
    sum_r = np.float64(1.)
    active_nodes = 0
    epoch_num = 0
    freq_com = np.zeros(n, dtype=np.float64)

    l1_error = []
    num_oper_list = []
    step=1e4
    threshold = step

    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:  # the end of t-th epoch.
            l1_error.append(sum_r)
            num_oper_list.append(num_oper)
            epoch_num += 1
            # add ddag into the queue
            rear = (rear + 1) % n
            queue[rear] = n
            active_nodes = 0
            continue
        active_nodes += 1
        freq_com[u] += 1
        num_oper += degree[u]

        res_u = r[u]
        sum_r -= omega * (alpha * res_u)
        
        if num_oper>threshold:
            l1_error.append(sum_r)
            num_oper_list.append(num_oper)
            threshold += step

        gpr_vec[u] += omega * (alpha * res_u)
        r[u] -= omega * r[u]
        push_amount = (res_u * (1. - alpha) * omega) / degree[u]
        for v in indices[indptr[u]:indptr[u + 1]]:
            r[v] += push_amount
            if not q_mark[v] and np.abs(r[v]) >= eps_vec[v]:
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True

    l1_error.append(sum_r)
    num_oper_list.append(num_oper)
    epoch_num=len(l1_error)

    re = np.zeros(n + 2*epoch_num +1, dtype=np.float64)
    re[:n] = gpr_vec
    re[n:n+epoch_num] = l1_error
    re[n+epoch_num:-1] = num_oper_list
    re[-1]= epoch_num
    return re


def forward_push_sor(adj_matrix,s,eps,alpha,omega=None):
    alpha=1-alpha
    if not omega:
        omega = 1. + ((1. - alpha) / (1. + np.sqrt(1 - (1. - alpha) ** 2.))) ** 2
    # omega=1.2
    
    degree=np.int32(adj_matrix.sum(1).A.flatten())
    indices=adj_matrix.indices
    indptr=adj_matrix.indptr
    eps=eps/indptr[-1]
    
    vec_num_op=_forward_push_sor(indptr,indices,degree,s,eps,alpha,omega)

    n=adj_matrix.shape[0]
    epoch_num=int(vec_num_op[-1])
    gpr_vec=vec_num_op[:n]
    l1_error=vec_num_op[n:n+epoch_num]
    num_oper_list=vec_num_op[n+epoch_num:-1]
    return gpr_vec,num_oper_list,l1_error


@njit(cache=True)
def _power_push_sor(indptr,indices,degree,s,eps,alpha,omega):
    
    n = degree.shape[0]
    m = indptr[-1]
    queue = np.zeros(n, dtype=np.int64)
    front, rear = np.int64(0), np.int64(1)

    gpr_vec, res = np.zeros(n), np.zeros(n)
    queue[rear] = s
    res[s] = 1
    q_mark = np.zeros(n)
    q_mark[s] = 1
    switch_size = np.int64(n / 4)
    r_max = eps / m
    r_sum = 1
    eps_vec = r_max * degree

    num_oper=0.
    l1_error = []
    num_oper_list = []
    step=1e5
    threshold = step

    while front != rear and ((rear - front) <= switch_size):
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if np.abs(res[u]) > eps_vec[u]:
            residual = omega * alpha * res[u]
            gpr_vec[u] += residual
            r_sum -= residual
            increment = omega * (1. - alpha) * res[u] / degree[u]
            res[u] -= omega * res[u]
            num_oper += degree[u]

            if num_oper>threshold:
                l1_error.append(r_sum)
                num_oper_list.append(num_oper)
                threshold += step
                if threshold>3e7:
                    step=1e5

            for v in indices[indptr[u]:indptr[u + 1]]:
                res[v] += increment
                if not q_mark[v]:
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True
    jump = False
    if r_sum <= eps:
        jump = True
    num_epoch = 8

    if not jump:
        for epoch in np.arange(1, num_epoch + 1):
            r_max_prime1 = np.power(eps, epoch / num_epoch)
            r_max_prime2 = np.power(eps, epoch / num_epoch) / m
            while r_sum > r_max_prime1:
                u = 0
                while u < n:
                    if np.abs(res[u]) > r_max_prime2 * degree[u]:
                        residual = omega * alpha * res[u]
                        gpr_vec[u] += residual
                        r_sum -= residual
                        increment = omega * (1. - alpha) * res[u] / degree[u]
                        res[u] -= omega * res[u]
                        num_oper += degree[u]

                        if num_oper>threshold:
                            l1_error.append(r_sum)
                            num_oper_list.append(num_oper)
                            threshold += step
                            if threshold>3e7:
                                step=1e5

                        for index in indices[indptr[u]:indptr[u + 1]]:
                            res[index] += increment
                    u += 1

    l1_error.append(r_sum)
    num_oper_list.append(num_oper)
    epoch_num=len(l1_error)

    re = np.zeros(n + 2*epoch_num +1, dtype=np.float64)
    re[:n] = gpr_vec
    re[n:n+epoch_num] = l1_error
    re[n+epoch_num:-1] = num_oper_list
    re[-1]= epoch_num
    return re

def power_push_sor(adj_matrix,s,eps,alpha,omega=None):
    alpha=1-alpha
    if not omega:
        omega = 1. + ((1. - alpha) / (1. + np.sqrt(1 - (1. - alpha) ** 2.))) ** 2

    degree=np.int32(adj_matrix.sum(1).A.flatten())
    indices=adj_matrix.indices
    indptr=adj_matrix.indptr
    eps=eps/indptr[-1]
    
    vec_num_op=_power_push_sor(indptr,indices,degree,s,eps,alpha,omega)

    n=adj_matrix.shape[0]
    epoch_num=int(vec_num_op[-1])
    gpr_vec=vec_num_op[:n]
    l1_error=vec_num_op[n:n+epoch_num]
    num_oper_list=vec_num_op[n+epoch_num:-1]
    return gpr_vec,num_oper_list,l1_error