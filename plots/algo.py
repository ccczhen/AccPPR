import networkx as nx
import numpy as np
import numba
import scipy.linalg
from numba import jit
import scipy.sparse as sp
from numba import njit, float64, int64, int32, prange
import time


@njit(cache=True)
def _forward_push(indptr, indices, degree, s, eps, alpha):
    n = degree.shape[0]
    queue = np.zeros(n, dtype=np.int32)
    front, rear = np.int32(0), np.int32(1)
    gpr_vec = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    queue[rear] = s
    r[s] = 1.
    q_mark = np.zeros(n)
    q_mark[s] = 1
    eps_vec = np.zeros(n)
    for i in prange(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 0.
    sum_r = 1.
    while front != rear:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        num_oper += degree[u]
        res_u = r[u]
        sum_r -= alpha * res_u
        gpr_vec[u] += alpha * res_u
        rest_prob = (1. - alpha) / degree[u]
        r[u] = 0
        push_amount = rest_prob * res_u
        for v in indices[indptr[u]:indptr[u + 1]]:
            r[v] += push_amount
            if not q_mark[v] and np.abs(r[v]) > eps_vec[v]:
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True
    vec_num_op = np.zeros(2 * n + 1, np.float64)
    vec_num_op[:n] = gpr_vec
    vec_num_op[n:2 * n] = r
    vec_num_op[-1] = num_oper
    print(np.sum(gpr_vec), sum_r, num_oper)
    return vec_num_op


@njit(cache=True)
def _forward_push_dummy(indptr, indices, degree, alpha, eps, s):
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int64)
    front, rear = np.int64(0), np.int64(2)
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = s
    queue[2] = n  # dummy node
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    q_mark[n] = 1
    eps_vec = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    # debug
    num_oper = np.float64(0.)
    sum_r = np.float64(1.)
    active_nodes = 0
    epoch_num = 0
    freq_com = np.zeros(n, dtype=np.float64)
    l1_error = []
    num_oper_list = [0]
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
    vec_num_op = np.zeros(n + 3)
    vec_num_op[:n] = gpr_vec
    vec_num_op[n] = num_oper
    vec_num_op[n+1] = sum_r
    vec_num_op[n+2] = epoch_num
    return vec_num_op


@njit(cache=True)
def _forward_push_dummy_sor(indptr, indices, degree, alpha, eps, s, omega):
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int64)
    front, rear = np.int64(0), np.int64(2)
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = s
    queue[2] = n  # dummy node
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    q_mark[n] = 1
    eps_vec = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    # debug
    num_oper = np.float64(0.)
    sum_r = np.float64(1.)
    active_nodes = 0
    epoch_num = 0
    freq_com = np.zeros(n, dtype=np.float64)
    l1_error = []
    num_oper_list = [0]
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
            if np.abs(sum_r) > 1e3:
                break
            continue
        active_nodes += 1
        freq_com[u] += 1
        num_oper += degree[u]
        res_u = r[u]
        gpr_vec[u] += omega * (alpha * res_u)
        sum_r -= omega * (alpha * res_u)
        r[u] -= omega * r[u]
        push_amount = (res_u * (1. - alpha) * omega) / degree[u]

        for v in indices[indptr[u]:indptr[u + 1]]:
            r[v] += push_amount
            if not q_mark[v] and np.abs(r[v]) >= eps_vec[v]:
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True
        if np.abs(sum_r) > 1e3:
            break
    vec_num_op = np.zeros(n + 2)
    vec_num_op[:-2] = gpr_vec
    vec_num_op[-2] = num_oper
    vec_num_op[-1] = sum_r
    return vec_num_op


@njit(cache=True)
def _gaussian_seidel_sor(indptr, indices, degree, alpha, eps, s, omega):
    n = len(degree)
    m = len(indices)
    phi = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    b[s] = alpha
    sum_phi = 0.
    num_oper = 0
    while np.abs(1. - sum_phi) > (eps * m):
        for u in range(n):
            sigma = 0
            for v in indices[indptr[u]:indptr[u + 1]]:
                sigma += -((1. - alpha) / degree[v]) * phi[v]
                num_oper += degree[u]
            tmp = (1. - omega) * phi[u] + omega * (b[u] - sigma)
            sum_phi += (tmp - phi[u])
            phi[u] = tmp
    return phi


@njit(cache=True)
def _power_push(indptr, indices, degree, s, eps, alpha):
    n = degree.shape[0]
    m = indptr[-1]
    queue = np.zeros(n, dtype=np.int64)
    front, rear = np.int64(0), np.int64(1)
    gpr_vec = np.zeros(n, dtype=np.float64)
    res = np.zeros(n, dtype=np.float64)
    queue[rear] = s
    res[s] = 1.
    q_mark = np.zeros(n, dtype=np.bool_)
    q_mark[s] = 1.
    switch_size = np.int64(n / 4)
    r_max = eps / m
    r_sum = 1
    eps_vec = r_max * degree

    num_oper = 0.
    while front != rear and ((rear - front) <= switch_size):
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if np.abs(res[u]) > eps_vec[u]:
            alpha_residual = (1 - alpha) * res[u]

            gpr_vec[u] += alpha_residual
            r_sum -= alpha_residual
            increment = (res[u] - alpha_residual) / degree[u]
            res[u] = 0
            num_oper += degree[u]
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
                    if res[u] > r_max_prime2 * degree[u]:
                        alpha_residual = (1 - alpha) * res[u]
                        gpr_vec[u] += alpha_residual
                        r_sum -= alpha_residual
                        increment = (res[u] - alpha_residual) / degree[u]
                        res[u] = 0
                        num_oper += degree[u]
                        for index in indices[indptr[u]:indptr[u + 1]]:
                            res[index] += increment
                    u += 1

    vec_num_op = np.zeros(n + 1)
    vec_num_op[:-1] = gpr_vec
    vec_num_op[-1] = num_oper
    return vec_num_op


@njit(cache=True)
def _power_push_sor(indptr, indices, degree, s, eps, alpha, omega):
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
    num_oper = 0.
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
            for v in indices[indptr[u]:indptr[u + 1]]:
                res[v] += increment
                if not q_mark[v]:
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True
            if np.abs(r_sum) > 1e3:
                break
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
                        for index in indices[indptr[u]:indptr[u + 1]]:
                            res[index] += increment
                    u += 1
                if np.abs(r_sum) > 1e3:
                    break
    vec_num_op = np.zeros(n + 1)
    vec_num_op[:-1] = gpr_vec
    vec_num_op[-1] = num_oper
    return vec_num_op


@njit(cache=True)
def _forward_push_dummy_bound(indptr, indices, degree, alpha, eps, s):
    """
    This is a showcase of dummy node. It is equivalent to forward_push_normalized.
    :param s: the source node s
    :return:
    """
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int32)
    front = np.int32(0)
    rear = np.int32(2)
    ppr_vec = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    queue[1] = s
    queue[rear] = n  # ddag, the dummy node in our paper
    r[s] = 1.
    q_mark = np.zeros(n + 1, dtype=np.bool_)
    q_mark[s] = True
    q_mark[n] = True
    eps_vec = np.zeros(n, dtype=np.float64)
    t = 0
    for i in np.arange(n):
        eps_vec[i] = degree[i] * eps

    vol_active = 0
    vol_total = degree[s]
    aver_active_vol = 0.
    ratio_list = []

    num_operations = 0.
    sum_r = 1.
    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:
            # the last epoch may not have any active nodes
            if vol_active <= 0.:
                rear = (rear + 1) % n
                queue[rear] = n
                q_mark[n] = True
                continue
            aver_active_vol += vol_active
            ratio_list.append(vol_active / vol_total)
            vol_active = 0.
            vol_total = 0.
            for _ in range(n):
                if r[_] != 0.:
                    vol_total += degree[_]
            t = t + 1
            rear = (rear + 1) % n
            queue[rear] = n
            q_mark[n] = True
            continue
        num_operations += degree[u]
        vol_active += degree[u]
        res_u = r[u]
        ppr_vec[u] += alpha * res_u
        sum_r -= alpha * res_u
        rest_prob = (1. - alpha) / degree[u]
        r[u] = 0
        push_amount = rest_prob * res_u
        for v in indices[indptr[u]:indptr[u + 1]]:
            r[v] += push_amount
            if not q_mark[v] and (r[v] >= eps_vec[v]):
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True

    aver_ratio = 0.
    for _ in ratio_list:
        aver_ratio += _
    if t > 0:
        aver_ratio /= t
        aver_active_vol /= t
    st = 0
    for i in range(n):
        if r[i] != 0.:
            st += 1
    theoretical_bound = 0.
    estimate_t = 0
    if aver_ratio > 0.:
        theoretical_bound = (aver_active_vol / (alpha * aver_ratio)) * np.log(1. / (eps * (1. - alpha) * st))
        estimate_t = (1. / (alpha * aver_ratio)) * np.log(1. / (eps * (1. - alpha) * st))
    if num_operations > theoretical_bound:
        print(r"error", num_operations, theoretical_bound)
    return sum_r, aver_active_vol, aver_ratio, t, estimate_t, num_operations, theoretical_bound, ppr_vec


@njit(cache=True)
def _forward_push_mean(indptr, indices, degree, alpha, eps, s, debug=True):
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
    num_oper_list = [0]
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
            # due to precision problem, we should add a small constant
            mean_r -= 1e-15
            l1_error.append(sum_r)
            num_oper_list.append(num_oper)
            epoch_num += 1
            # add ddag into the queue
            rear = (rear + 1) % n
            queue[rear] = n
            if debug:
                print(active_nodes, n, num_oper - num_oper_list[-2], len(indices))
            active_nodes = 0
            continue
        # it is not worth to push node u
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
    re = np.zeros(n + 3 + 2 * len(l1_error), dtype=np.float64)
    re[:n] = freq_com
    re[n] = num_oper
    re[n + 1] = sum_r
    re[n + 2] = epoch_num
    re[n + 3:n + 3 + len(l1_error)] = l1_error
    re[n + 3 + len(l1_error):] = num_oper_list[1:]
    return re


@njit(cache=True)
def _forward_push_mean_old(indptr, indices, degree, alpha, eps, s, debug=True):
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
    mean_deg = 0.
    for i in range(n):
        eps_vec[i] = degree[i] * eps
        mean_deg += degree[i]
    mean_deg /= n

    # debug
    num_oper = np.float64(0.)
    sum_r = np.float64(1.)
    active_nodes = 0
    epoch_num = 0
    freq_com = np.zeros(n, dtype=np.float64)
    l1_error = []
    num_oper_list = [0]
    mean_r = np.infty
    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:
            if active_nodes >= (n / mean_deg):
                mean_r = 0.
                non_zero = 0
                for i in range(n):
                    if r[i] >= eps_vec[i]:
                        mean_r += r[i] / degree[i]
                        non_zero += 1
                mean_r /= non_zero
                mean_r += 1e-15  # due to precision problem, we should add a small constant
            else:
                mean_r = 0.

            l1_error.append(sum_r)
            num_oper_list.append(num_oper)
            epoch_num += 1
            # add ddag into the queue
            rear = (rear + 1) % n
            queue[rear] = n
            if debug:
                print(active_nodes, n, num_oper - num_oper_list[-2], len(indices))
            active_nodes = 0
            continue
        # it is not worth to push node u
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
    re = np.zeros(n + 3 + 2 * len(l1_error), dtype=np.float64)
    re[:n] = freq_com
    re[n] = num_oper
    re[n + 1] = sum_r
    re[n + 2] = epoch_num
    re[n + 3:n + 3 + len(l1_error)] = l1_error
    re[n + 3 + len(l1_error):] = num_oper_list[1:]
    for i in prange(n):
        if r[i] > eps_vec[i]:
            print(i, 'error', r[i], eps_vec[i])
    return re


def gpr_forward_push_simultaneous(csr_mat: sp.csr_matrix, r: np.array, alpha: np.float64, max_iter: int, debug=False):
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    n, _ = csr_mat.shape
    p = np.zeros(n, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    deg = np.array(csr_mat.sum(axis=1)).flatten()
    num_pushes = 0
    results = []
    num_iter = 0
    while num_iter < max_iter:
        if debug:
            results.append(np.copy(p))
            num_iter += 1
            if num_iter >= max_iter:
                break
        for uu in range(n):
            if r[uu] <= 0. or deg[uu] <= 0.:
                continue
            res = r[uu]
            p[uu] += alpha * res
            rest_prob = (1. - alpha) * res / deg[uu]
            r[uu] = 0.  # put it first to avoid the self-loop trap.
            num_pushes += 1
            for vv in indices[indptr[uu]:indptr[uu + 1]]:
                r[vv] += rest_prob
    return p, r, num_pushes, results


def gpr_forward_push_asynchronous(csr_mat: sp.csr_matrix, r: np.array, alpha: np.float64, max_iter: int, debug=False):
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    n, _ = csr_mat.shape
    p = np.zeros(n, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    deg = np.array(csr_mat.sum(axis=1)).flatten()
    num_pushes = 0
    results = []
    num_iter = 0
    new_r = np.zeros_like(r)
    while num_iter < max_iter:
        if debug:
            results.append(np.copy(p))
            num_iter += 1
            if num_iter >= max_iter:
                break
        for uu in range(n):
            if r[uu] <= 0. or deg[uu] <= 0.:
                continue
            res = r[uu]
            p[uu] += alpha * res
            rest_prob = (1. - alpha) * res / deg[uu]
            r[uu] = 0.  # put it first to avoid the self-loop trap.
            num_pushes += 1
            for vv in indices[indptr[uu]:indptr[uu + 1]]:
                new_r[vv] += rest_prob
        r, new_r = new_r, r
    return p, r, num_pushes, results


@njit(cache=True)
def forward_push_r_deg_mean(indptr, indices, degree, alpha, eps, s):
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int64)
    front, rear = 0, 2
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = n
    queue[rear] = s  # a funny flag to denote an epoch
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    eps_vec = np.zeros(n + 1)
    mean_deg = len(indices) / n
    for i in range(n):
        eps_vec[i] = degree[i] * eps

    num_oper = 0.
    sum_r = 1.
    active_nodes = 1
    epoch_num = 0
    while True:
        if active_nodes <= 0:
            break
        # if queue no active nodes, exit
        # next epoch number
        epoch_num += 1
        # calculate the mean of active nodes
        g_mean = 0.
        for i in prange(n):
            if r[i] > eps_vec[i]:
                g_mean += r[i] / degree[i]
        g_mean /= active_nodes
        active_nodes = 0
        if active_nodes <= n / mean_deg:
            while front != rear:
                front = (front + 1) % n
                u = queue[front]
                q_mark[u] = False
                if r[u] > eps_vec[u]:
                    num_oper += degree[u]
                    res_u = r[u]
                    sum_r -= alpha * res_u
                    gpr_vec[u] += alpha * res_u
                    rest_prob = (1. - alpha) / degree[u]
                    r[u] = 0
                    push_amount = rest_prob * res_u
                    for v in indices[indptr[u]:indptr[u + 1]]:
                        r[v] += push_amount
                        if not q_mark[v]:
                            rear = (rear + 1) % n
                            queue[rear] = v
                            q_mark[v] = True
                            active_nodes += 1
        else:
            r_est = r[u] / degree[u]
            if r[u] > eps_vec[u] and (r_est >= g_mean):
                num_oper += degree[u]
                res_u = r[u]
                sum_r -= alpha * res_u
                gpr_vec[u] += alpha * res_u
                rest_prob = (1. - alpha) / degree[u]
                r[u] = 0
                push_amount = rest_prob * res_u
                for v in indices[indptr[u]:indptr[u + 1]]:
                    r[v] += push_amount
                    if not q_mark[v]:
                        rear = (rear + 1) % n
                        queue[rear] = v
                        q_mark[v] = True
                        active_nodes += 1
            elif r[u] > eps_vec[u] and (r_est < g_mean):
                # move to next S_t
                rear = (rear + 1) % n
                queue[rear] = u
                q_mark[u] = True
                active_nodes += 1
    re = [num_oper, sum_r, epoch_num]
    for i in prange(n):
        if r[i] > eps_vec[i]:
            print('error')
    return re


def forward_push_r_deg_mean(indptr, indices, degree, alpha, eps, s):
    start_time = time.time()
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int64)
    front, rear = 0, 2
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = n
    queue[rear] = s  # a funny flag to denote an epoch
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    eps_vec = np.zeros(n + 1)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 0.
    sum_r = 1.
    count_r_nonzero = []
    count_active_nodes = []
    count_num_operations = []
    count_r_sum = []
    count_r_sum_power_iter = []
    active_nodes = 0
    count_num_operations_epoch = []
    num_oper_epoch = 0
    epoch_num = 0
    g_mean = 0.
    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:
            epoch_num += 1
            count_r_sum.append(sum_r)
            count_r_sum_power_iter.append((1 - alpha) ** epoch_num)
            # push to the queue again
            rear = (rear + 1) % n
            queue[rear] = n
            count_r_nonzero.append(np.count_nonzero(r))
            count_active_nodes.append(active_nodes)
            count_num_operations.append(num_oper)
            count_num_operations_epoch.append(num_oper_epoch)
            active_nodes = 0
            num_oper_epoch = 0
            x = [r[_] / degree[_] for _ in range(n) if r[_] > eps_vec[_]]
            if len(x) <= 0:
                break
            g_mean = np.mean(x)

        if np.abs(r[u]) > eps_vec[u] and (r[u] / degree[u] >= g_mean):
            active_nodes += 1
            num_oper_epoch += degree[u]
            num_oper += degree[u]
            res_u = r[u]
            sum_r -= alpha * res_u
            gpr_vec[u] += alpha * res_u
            rest_prob = (1. - alpha) / degree[u]
            r[u] = 0
            push_amount = rest_prob * res_u
            for v in indices[indptr[u]:indptr[u + 1]]:
                r[v] += push_amount
                if not q_mark[v]:
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True
        elif np.abs(r[u]) > eps_vec[u] and (r[u] / degree[u] < g_mean):
            rear = (rear + 1) % n
            queue[rear] = u
            q_mark[u] = True

    count_r_sum_power_iter = [_ for _ in count_r_sum_power_iter if _ > sum_r]
    results = {"count_r_nonzero": count_r_nonzero,
               "count_active_nodes": count_active_nodes,
               "count_num_operations": count_num_operations,
               "count_r_sum_power_iter": count_r_sum_power_iter,
               "count_num_operations_epoch": count_num_operations_epoch}
    print(f"finish in {time.time() - start_time:.3e} with operations: "
          f"{num_oper:.0f} r-sum:{sum_r:.3e} ep-num: {epoch_num:.0f}")
    return results


@njit(cache=True)
def forward_push_r_deg_mean_v0(indptr, indices, degree, alpha, eps, s):
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int64)
    front, rear = 0, 2
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = n
    queue[rear] = s  # a funny flag to denote an epoch
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    eps_vec = np.zeros(n + 1)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 1.
    sum_r = 1.
    active_nodes = 1
    epoch_num = 0
    g_mean = 0.

    num_act = 0
    num_act_work = 0
    num_inact = 0
    num_inact_work = 0
    freq_com = np.zeros(n)
    l1_error = []
    num_oper_list = []
    while True:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:
            l1_error.append(sum_r)
            num_oper_list.append(num_oper)
            # if queue no active nodes, exit
            if active_nodes <= 0:
                break
            # next epoch number
            epoch_num += 1
            # calculate the mean of active nodes
            g_mean = 0.
            for i in prange(n):
                if r[i] > eps_vec[i]:
                    g_mean += r[i] / degree[i]
            g_mean /= active_nodes
            # push the funny flag to the queue again
            rear = (rear + 1) % n
            queue[rear] = n
            active_nodes = 0

            # print(num_act, num_act_work, num_inact, num_inact_work)
            num_act = 0
            num_act_work = 0
            num_inact = 0
            num_inact_work = 0
            continue
        r_est = r[u] / degree[u]

        if r[u] > eps_vec[u] and (r_est >= g_mean):
            num_act += 1
            num_act_work += degree[u]
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
                if not q_mark[v]:
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True
                    active_nodes += 1
        elif r[u] > eps_vec[u] and (r_est < g_mean):
            num_inact += 1
            num_inact_work += degree[u]
            # move to next S_t
            rear = (rear + 1) % n
            queue[rear] = u
            q_mark[u] = True
            active_nodes += 1
    for i in prange(n):
        if r[i] > eps_vec[i]:
            print('error')
    re = np.zeros(n + 3 + 2 * len(l1_error))
    re[:n] = freq_com
    re[n] = num_oper
    re[n + 1] = sum_r
    re[n + 2] = epoch_num
    re[n + 3:n + 3 + len(l1_error)] = l1_error
    re[n + 3 + len(l1_error):] = num_oper_list
    return re


@njit(cache=True)
def forward_push_r_deg_mean_v1(indptr, indices, degree, alpha, eps, s):
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int64)
    front, rear = 0, 2
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = n
    queue[rear] = s  # a funny flag to denote an epoch
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    eps_vec = np.zeros(n + 1)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 1.
    sum_r = 1.
    active_nodes = 1
    epoch_num = 0
    g_mean = 0.
    freq_com = np.zeros(n)
    l1_error = []
    num_oper_list = []
    while True:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:
            l1_error.append(sum_r)
            num_oper_list.append(num_oper)
            # if queue no active nodes, exit
            if active_nodes <= 0:
                break
            # next epoch number
            epoch_num += 1
            # calculate the mean of active nodes
            g_mean = 0.
            num_nonzero = 0
            for i in prange(n):
                if r[i] > 0:
                    g_mean += r[i] / degree[i]
                    num_nonzero += 1
            g_mean /= num_nonzero
            # push the funny flag to the queue again
            rear = (rear + 1) % n
            queue[rear] = n
            active_nodes = 0
            continue
        r_est = r[u] / degree[u]
        if r[u] > eps_vec[u] and (r_est >= g_mean):
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
                if not q_mark[v]:
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True
                    active_nodes += 1
        elif r[u] > eps_vec[u] and (r_est < g_mean):
            # move to next S_t
            rear = (rear + 1) % n
            queue[rear] = u
            q_mark[u] = True
            active_nodes += 1
    for i in prange(n):
        if r[i] > eps_vec[i]:
            print('error')
    re = np.zeros(n + 3 + 2 * len(l1_error))
    re[:n] = freq_com
    re[n] = num_oper
    re[n + 1] = sum_r
    re[n + 2] = epoch_num
    re[n + 3:n + 3 + len(l1_error)] = l1_error
    re[n + 3 + len(l1_error):] = num_oper_list
    return re


@njit(cache=True)
def forward_push_r_deg_mean_v2(indptr, indices, degree, alpha, eps, s):
    n = len(degree)
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    r[s] = 1.
    eps_vec = np.zeros(n + 1)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 1.
    sum_r = 1.
    active_nodes = 1
    epoch_num = 0
    g_mean = 0.
    freq_com = np.zeros(n)
    l1_error = []
    num_oper_list = []
    while active_nodes > 0:
        l1_error.append(sum_r)
        num_oper_list.append(num_oper)
        # if queue no active nodes, exit
        if active_nodes <= 0:
            break
        # next epoch number
        epoch_num += 1
        # calculate the mean of active nodes
        g_mean = 0.
        num_nonzero = 0
        for i in prange(n):
            if r[i] > 0:
                g_mean += r[i] / np.sqrt(degree[i])
                num_nonzero += 1
        g_mean /= num_nonzero
        active_nodes = 0
        continue
        r_est = r[u] / np.sqrt(degree[u])
        if r[u] > eps_vec[u] and (r_est >= g_mean):
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
                if not q_mark[v]:
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True
                    active_nodes += 1
        elif r[u] > eps_vec[u] and (r_est < g_mean):
            # move to next S_t
            rear = (rear + 1) % n
            queue[rear] = u
            q_mark[u] = True
            active_nodes += 1
    for i in prange(n):
        if r[i] > eps_vec[i]:
            print('error')
    re = np.zeros(n + 3 + 2 * len(l1_error))
    re[:n] = freq_com
    re[n] = num_oper
    re[n + 1] = sum_r
    re[n + 2] = epoch_num
    re[n + 3:n + 3 + len(l1_error)] = l1_error
    re[n + 3 + len(l1_error):] = num_oper_list
    return re


def forward_push_r_deg_mean_2(indptr, indices, degree, alpha, eps, s, mean_type):
    start_time = time.time()
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int64)
    front, rear = 0, 2
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = n
    queue[rear] = s  # a funny flag to denote an epoch
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    eps_vec = np.zeros(n + 1)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 0.
    sum_r = 1.
    count_r_nonzero = []
    count_active_nodes = []
    count_num_operations = []
    count_r_sum = []
    count_r_sum_power_iter = []
    active_nodes = 0
    count_num_operations_epoch = []
    num_oper_epoch = 0
    epoch_num = 0
    g_mean = 0.
    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:
            epoch_num += 1
            count_r_sum.append(sum_r)
            count_r_sum_power_iter.append((1 - alpha) ** epoch_num)
            # push to the queue again
            rear = (rear + 1) % n
            queue[rear] = n
            count_r_nonzero.append(np.count_nonzero(r))
            count_active_nodes.append(active_nodes)
            count_num_operations.append(num_oper)
            count_num_operations_epoch.append(num_oper_epoch)
            active_nodes = 0
            num_oper_epoch = 0
            x = [r[_] / degree[_] for _ in range(n) if r[_] > eps_vec[_]]
            if len(x) <= 0:
                break
            if mean_type == 0:
                g_mean = np.mean(x)
                xx = [_ for _ in x if _ >= g_mean]
                yy = [_ for _ in x if _ < g_mean]
                work1 = [degree[_] for _ in range(n)
                         if r[_] / degree[_] >= g_mean]
                work2 = [degree[_] for _ in range(n)
                         if (r[_] / degree[_] < g_mean and r[_] > eps_vec[_])]
                print(epoch_num, np.sum(r), np.sum(x), np.sum(xx),
                      np.sum(yy), np.sum(xx) / np.sum(yy),
                      np.sum(work1), np.sum(work2), np.sum(work1) / np.sum(work2))
                # split of work1
                work11 = [_ for _ in range(n)
                          if (r[_] / degree[_] >= g_mean and
                              (degree[_] <= np.mean(work1)))]
                total_work = [_ for _ in range(n) if r[_] > eps_vec[_]]
                print('key')
                print(np.sum([r[_] for _ in work11]) /
                      np.sum([r[_] for _ in total_work]))
                print(np.sum([degree[_] for _ in range(n) if r[_] > eps_vec[_]]) /
                      np.sum([degree[_] for _ in work11]))
            else:
                g_mean = gmean(x)
        if rear - front < n / 4:
            if np.abs(r[u]) > eps_vec[u]:
                active_nodes += 1
                num_oper_epoch += degree[u]
                num_oper += degree[u]
                res_u = r[u]
                sum_r -= alpha * res_u
                gpr_vec[u] += alpha * res_u
                rest_prob = (1. - alpha) / degree[u]
                r[u] = 0
                push_amount = rest_prob * res_u
                for v in indices[indptr[u]:indptr[u + 1]]:
                    r[v] += push_amount
                    if not q_mark[v]:
                        rear = (rear + 1) % n
                        queue[rear] = v
                        q_mark[v] = True
        else:
            if np.abs(r[u]) > eps_vec[u] and (r[u] / degree[u] >= g_mean):
                active_nodes += 1
                num_oper_epoch += degree[u]
                num_oper += degree[u]
                res_u = r[u]
                sum_r -= alpha * res_u
                gpr_vec[u] += alpha * res_u
                rest_prob = (1. - alpha) / degree[u]
                r[u] = 0
                push_amount = rest_prob * res_u
                for v in indices[indptr[u]:indptr[u + 1]]:
                    r[v] += push_amount
                    if not q_mark[v]:
                        rear = (rear + 1) % n
                        queue[rear] = v
                        q_mark[v] = True
            elif np.abs(r[u]) > eps_vec[u] and (r[u] / degree[u] < g_mean):
                rear = (rear + 1) % n
                queue[rear] = u
                q_mark[u] = True

    count_r_sum_power_iter = [_ for _ in count_r_sum_power_iter if _ > sum_r]
    results = {"count_r_nonzero": count_r_nonzero,
               "count_active_nodes": count_active_nodes,
               "count_num_operations": count_num_operations,
               "count_r_sum_power_iter": count_r_sum_power_iter,
               "count_num_operations_epoch": count_num_operations_epoch}
    print(f"finish in {time.time() - start_time} with operations: {num_oper} r-sum:{sum_r}")
    print(f"{epoch_num}")
    print(f"error ! r[i]:{len([r[_] for _ in range(n) if r[_] > eps_vec[_]])}")
    return results


def forward_push_original_debug(indptr, indices, degree, alpha, eps, s):
    start_time = time.time()
    n = len(degree)
    print(n)
    queue = np.zeros(n + 1, dtype=np.int64)
    front, rear = 0, 2
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = n
    queue[rear] = s  # a funny flag to denote an epoch
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    eps_vec = np.zeros(n + 1)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 0.
    sum_r = 1.
    active_nodes = 0
    num_oper_epoch = 0
    epoch_num = 0
    g_mean = 0.
    act_deg_sum = np.log(degree[s])
    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:  # at the initial
            epoch_num += 1
            active_nodes = 0
            num_oper_epoch = 0
            g_mean = np.exp(act_deg_sum / (rear - front))
            print('--: ', epoch_num, rear, front, g_mean)

            act_nodes = [_ for _ in range(n) if r[_] > eps_vec[_]]
            if len(act_nodes) == 0:
                continue
            act_degrees = [degree[_] for _ in act_nodes]
            aver_act_deg = np.mean(act_degrees)
            low_nodes = [_ for _ in act_nodes if degree[_] <= aver_act_deg]
            low_nodes_deg = [degree[_] for _ in act_nodes if degree[_] <= aver_act_deg]
            high_nodes = [_ for _ in act_nodes if degree[_] > aver_act_deg]
            high_nodes_deg = [degree[_] for _ in act_nodes if degree[_] > aver_act_deg]
            r_mean_low = np.mean([r[_] for _ in low_nodes])
            r_mean_high = np.mean([r[_] for _ in high_nodes])
            print(epoch_num)
            print("mean-lower nodes", np.sum(low_nodes_deg), len(low_nodes), r_mean_low,
                  np.sum([r[_] for _ in low_nodes]))
            print("mean-high nodes", np.sum(high_nodes_deg), len(high_nodes), r_mean_high,
                  np.sum([r[_] for _ in high_nodes]))
            act_nodes = [_ for _ in range(n) if r[_] > eps_vec[_]]
            act_degrees = [degree[_] for _ in act_nodes]
            aver_act_deg = gmean(act_degrees)
            low_nodes = [_ for _ in act_nodes if degree[_] <= aver_act_deg]
            low_nodes_deg = [degree[_] for _ in act_nodes if degree[_] <= aver_act_deg]
            high_nodes = [_ for _ in act_nodes if degree[_] > aver_act_deg]
            high_nodes_deg = [degree[_] for _ in act_nodes if degree[_] > aver_act_deg]
            r_mean_low = np.mean([r[_] for _ in low_nodes])
            r_mean_high = np.mean([r[_] for _ in high_nodes])
            print("gmean-lower nodes", np.sum(low_nodes_deg), len(low_nodes), r_mean_low,
                  np.sum([r[_] for _ in low_nodes]))
            print("gmean-high nodes", np.sum(high_nodes_deg), len(high_nodes), r_mean_high,
                  np.sum([r[_] for _ in high_nodes]))
            print(f"{num_oper}")
            if (epoch_num - 3) % 3 == 0:
                print(f"next iteration for high nodes")
            else:
                print(f"next iteration for lower nodes")
            print('---')
            # time.sleep(1.5)

            act_deg_sum = 0.
            rear = (rear + 1) % n
            queue[rear] = n
            continue
        assert u != n
        if epoch_num <= 3:
            num_oper_epoch += degree[u]
            num_oper += degree[u]
            res_u = r[u]
            sum_r -= alpha * res_u
            gpr_vec[u] += alpha * res_u
            rest_prob = (1. - alpha) / degree[u]
            r[u] = 0
            push_amount = rest_prob * res_u
            for v in indices[indptr[u]:indptr[u + 1]]:
                r[v] += push_amount
                # nodes has not been put into queue yet
                if not q_mark[v] and (np.abs(r[v]) > eps_vec[v]):
                    active_nodes += 1
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True
                    act_deg_sum += np.log(degree[v])
        else:
            if (epoch_num - 3) % 5 != 0:  # processing lower degree nodes
                if degree[u] <= (g_mean + 2e-12):  # lower degree nodes
                    # print(epoch_num, degree[u], g_mean, num_oper, "process")
                    # time.sleep(0.01)
                    num_oper_epoch += degree[u]
                    num_oper += degree[u]
                    res_u = r[u]
                    sum_r -= alpha * res_u
                    gpr_vec[u] += alpha * res_u
                    rest_prob = (1. - alpha) / degree[u]
                    r[u] = 0
                    push_amount = rest_prob * res_u
                    for v in indices[indptr[u]:indptr[u + 1]]:
                        r[v] += push_amount
                        # nodes has not been put into queue yet
                        if not q_mark[v] and (np.abs(r[v]) > eps_vec[v]):
                            active_nodes += 1
                            rear = (rear + 1) % n
                            queue[rear] = v
                            q_mark[v] = True
                            act_deg_sum += np.log(degree[v])
                else:  # higher degree nodes, post for next time
                    act_deg_sum += np.log(degree[u])
                    rear = (rear + 1) % n
                    queue[rear] = u
                    q_mark[u] = True
                    # print(epoch_num, degree[u], g_mean, "ignored")
                    # time.sleep(0.01)
            else:  # processing high degree nodes
                if degree[u] > g_mean:  # higher degree nodes
                    num_oper_epoch += degree[u]
                    num_oper += degree[u]
                    res_u = r[u]
                    sum_r -= alpha * res_u
                    gpr_vec[u] += alpha * res_u
                    rest_prob = (1. - alpha) / degree[u]
                    r[u] = 0
                    push_amount = rest_prob * res_u
                    for v in indices[indptr[u]:indptr[u + 1]]:
                        r[v] += push_amount
                        if not q_mark[v] and (np.abs(r[v]) > eps_vec[v]):
                            active_nodes += 1
                            rear = (rear + 1) % n
                            queue[rear] = v
                            q_mark[v] = True
                            act_deg_sum += np.log(degree[v])
                else:  # lower degree nodes post for next time
                    act_deg_sum += np.log(degree[u])
                    rear = (rear + 1) % n
                    queue[rear] = u
                    q_mark[u] = True
    print(f"finish in {time.time() - start_time} with operations: {num_oper} r-sum:{sum_r}")
    print(f"error ! r[i]:{len([r[_] for _ in range(n) if r[_] > eps_vec[_]])}")


def forward_push_original_debug_2(indptr, indices, degree, alpha, eps, s):
    start_time = time.time()
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int32)
    front, rear = np.int32(0), np.int32(2)
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = s
    queue[rear] = n  # a funny flag to denote an epoch
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    eps_vec = np.zeros(n + 1)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 0.
    sum_r = 1.
    count_r_nonzero = []
    count_active_nodes = []
    count_num_operations = []
    count_r_sum = []
    count_r_sum_power_iter = []
    active_nodes = 0
    count_num_operations_epoch = []
    num_oper_epoch = 0
    epoch_num = 0
    r1 = []
    r2 = []
    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if np.abs(r[u]) > eps_vec[u]:
            active_nodes += 1
            num_oper_epoch += degree[u]
            num_oper += degree[u]
            res_u = r[u]
            sum_r -= alpha * res_u
            gpr_vec[u] += alpha * res_u
            rest_prob = (1. - alpha) / degree[u]
            r[u] = 0
            push_amount = rest_prob * res_u
            for v in indices[indptr[u]:indptr[u + 1]]:
                r[v] += push_amount
                if not q_mark[v]:
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True
        if u == n:

            epoch_num += 1
            count_r_sum.append(sum_r)
            count_r_sum_power_iter.append((1 - alpha) ** epoch_num)
            rear = (rear + 1) % n
            queue[rear] = n
            count_r_nonzero.append(np.count_nonzero(r))
            count_active_nodes.append(active_nodes)
            count_num_operations.append(num_oper)
            count_num_operations_epoch.append(num_oper_epoch)
            active_nodes = 0
            num_oper_epoch = 0

            act_nodes = [_ for _ in range(n) if r[_] > eps_vec[_]]
            if len(act_nodes) == 0:
                continue
            act_degrees = [degree[_] for _ in act_nodes]
            aver_act_deg = np.mean(act_degrees)
            low_nodes = [_ for _ in act_nodes if degree[_] <= aver_act_deg]
            low_nodes_deg = [degree[_] for _ in act_nodes if degree[_] <= aver_act_deg]
            high_nodes = [_ for _ in act_nodes if degree[_] > aver_act_deg]
            high_nodes_deg = [degree[_] for _ in act_nodes if degree[_] > aver_act_deg]
            r_mean_low = np.mean([r[_] for _ in low_nodes])
            r_mean_high = np.mean([r[_] for _ in high_nodes])
            print(epoch_num)
            print("mean-lower nodes", np.sum(low_nodes_deg), len(low_nodes), r_mean_low,
                  np.sum([r[_] for _ in low_nodes]))
            print("mean-high nodes", np.sum(high_nodes_deg), len(high_nodes), r_mean_high,
                  np.sum([r[_] for _ in high_nodes]))
            act_nodes = [_ for _ in range(n) if r[_] > eps_vec[_]]
            act_degrees = [degree[_] for _ in act_nodes]
            aver_act_deg = gmean(act_degrees)
            low_nodes = [_ for _ in act_nodes if degree[_] <= aver_act_deg]
            low_nodes_deg = [degree[_] for _ in act_nodes if degree[_] <= aver_act_deg]
            high_nodes = [_ for _ in act_nodes if degree[_] > aver_act_deg]
            high_nodes_deg = [degree[_] for _ in act_nodes if degree[_] > aver_act_deg]
            r_mean_low = np.mean([r[_] for _ in low_nodes])
            r_mean_high = np.mean([r[_] for _ in high_nodes])
            print("gmean-lower nodes", np.sum(low_nodes_deg), len(low_nodes), r_mean_low,
                  np.sum([r[_] for _ in low_nodes]))
            print("gmean-high nodes", np.sum(high_nodes_deg), len(high_nodes), r_mean_high,
                  np.sum([r[_] for _ in high_nodes]))
            print('---')


def forward_push_lower_degree(indptr, indices, degree, alpha, eps, s):
    start_time = time.time()
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int32)
    front, rear = np.int32(0), np.int32(2)
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = s
    queue[rear] = n  # a funny flag to denote an epoch
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    eps_vec = np.zeros(n + 1)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 0.
    sum_r = 1.
    count_r_nonzero = []
    count_active_nodes = []
    count_num_operations = []
    count_r_sum = []
    count_r_sum_power_iter = []
    active_nodes = 0
    count_num_operations_epoch = []
    num_oper_epoch = 0
    epoch_num = 0
    aver_deg = degree[s]
    total_deg = 0
    num_nodes = 0
    while (rear - front) > 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:
            epoch_num += 1
            count_r_sum.append(sum_r)
            count_r_sum_power_iter.append((1 - alpha) ** epoch_num)

            rear = (rear + 1) % n
            queue[rear] = n

            aver_deg = total_deg / num_nodes

            count_r_nonzero.append(np.count_nonzero(r))
            count_active_nodes.append(active_nodes)
            count_num_operations.append(num_oper)
            count_num_operations_epoch.append(num_oper_epoch)
            active_nodes = 0
            num_oper_epoch = 0
        elif degree[u] <= aver_deg:
            active_nodes += 1
            num_oper_epoch += degree[u]
            num_oper += degree[u]
            res_u = r[u]
            sum_r -= alpha * res_u
            gpr_vec[u] += alpha * res_u
            rest_prob = (1. - alpha) / degree[u]
            r[u] = 0
            push_amount = rest_prob * res_u
            for v in indices[indptr[u]:indptr[u + 1]]:
                r[v] += push_amount
                if not q_mark[v] and (np.abs(r[v]) > eps_vec[v]):
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True
                    total_deg += degree[v]
                    num_nodes += 1

    count_r_sum_power_iter = [_ for _ in count_r_sum_power_iter if _ > sum_r]
    results = {"count_r_nonzero": count_r_nonzero,
               "count_active_nodes": count_active_nodes,
               "count_num_operations": count_num_operations,
               "count_r_sum_power_iter": count_r_sum_power_iter,
               "count_num_operations_epoch": count_num_operations_epoch}
    print(f"finish in {time.time() - start_time} with operations: {num_oper} r-sum:{sum_r}")
    print(f"error ! r[i]:{len([r[_] for _ in range(n) if r[_] > eps_vec[_]])}")
    return results


def forward_push_mean(indptr, indices, degree, alpha, eps, s, factor):
    start_time = time.time()
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int32)
    front, rear = np.int32(0), np.int32(2)
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = s
    queue[rear] = n  # a funny flag to denote an epoch
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    eps_vec = np.zeros(n + 1)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 0.
    sum_r = 1.
    total_iter = 0
    count_r_nonzero = []
    count_active_nodes = []
    count_num_operations = []
    count_r_sum = []
    count_r_sum_power_iter = []
    active_nodes = 0
    count_num_operations_epoch = []
    num_oper_epoch = 0
    epoch_num = 0
    r_mean = 1.
    active_nodes = 1
    while True:
        u = queue[front]  # access front node
        if u == n:
            if active_nodes == 0:
                break
            # add into queue
            rear = (rear + 1) % n
            queue[rear] = n
            r_mean = np.mean([r[_] for _ in np.nonzero(r)[0] if r[_] > eps_vec[_]])

            epoch_num += 1
            count_r_sum.append(sum_r)
            count_r_sum_power_iter.append((1 - alpha) ** epoch_num)
            count_r_nonzero.append(np.count_nonzero(r))
            count_active_nodes.append(active_nodes)
            count_num_operations.append(num_oper)
            count_num_operations_epoch.append(num_oper_epoch)

            active_nodes = 0
            num_oper_epoch = 0
            continue
        if r[u] <= eps_vec[u]:
            # de-active node u
            front = (front + 1) % n
            q_mark[u] = False
            continue
        if np.abs(r[u]) >= r_mean:
            print(np.abs(r[u]), r_mean, u)
            # de-active node u
            front = (front + 1) % n
            q_mark[u] = False
            active_nodes += 1
            num_oper_epoch += degree[u]
            num_oper += degree[u]
            res_u = r[u]
            sum_r -= alpha * res_u
            gpr_vec[u] += alpha * res_u
            rest_prob = (1. - alpha) / degree[u]
            r[u] = 0
            push_amount = rest_prob * res_u
            for v in indices[indptr[u]:indptr[u + 1]]:
                r[v] += push_amount
                if not q_mark[v]:
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True
        total_iter += 1

    count_r_sum_power_iter = [_ for _ in count_r_sum_power_iter if _ > sum_r]
    results = {"count_r_nonzero": count_r_nonzero,
               "count_active_nodes": count_active_nodes,
               "count_num_operations": count_num_operations,
               "count_r_sum_power_iter": count_r_sum_power_iter,
               "count_num_operations_epoch": count_num_operations_epoch}
    print(f"finish in {time.time() - start_time} with operations: {num_oper} r-sum:{sum_r}")
    print(f"error ! r[i]:{len([_ for _ in r if _ > eps_vec[i]])}")
    return results


@numba.njit(cache=True, locals={'_val': numba.float64, 'res': numba.float64, 'res_vnode': numba.float64})
def _calc_ppr_node_dict(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f64_0 = numba.float64(0)
    p = {inode: f64_0}
    r = {inode: alpha}
    q = [inode]
    while len(q) > 0:
        unode = q.pop()  # TODO this is bad, should use q.pop(0)
        res = r[unode] if unode in r else f64_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f64_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val
            res_vnode = r[vnode] if vnode in r else f64_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)
    return list(p.keys()), list(p.values())


@jit(nopython=True)
def _calc_ppr_node_arr_stack(s, indptr, indices, deg, alpha, epsilon):
    f64_0 = numba.float64(0)
    n = numba.int32(len(indptr) - 1)
    p = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    alpha_deg = epsilon * np.asarray(deg, dtype=np.float64)
    r[s] = 1.
    stack = np.zeros(n, dtype=np.int32)
    head = numba.int32(0)
    stack_mark = np.zeros(n, dtype=numba.boolean)
    stack[head] = s
    head += 1
    stack_mark[s] = 1
    num_pushes = numba.int32(0)
    while head != 0:
        num_pushes += 1
        uu = stack[head - 1]
        head -= 1
        res = r[uu]
        p[uu] += alpha * res
        rest_prob = ((1. - alpha) * res / deg[uu])
        for vv in indices[indptr[uu]:indptr[uu + 1]]:
            r[vv] += rest_prob
            if r[vv] >= alpha_deg[vv]:
                if not stack_mark[vv]:
                    stack[head] = vv
                    head += 1
                    stack_mark[vv] = 1
        r[uu] = f64_0
        stack_mark[uu] = 0
    return p, r, num_pushes


@jit(nopython=True)
def _calc_ppr_node_arr_fifo(s, indptr, indices, deg, alpha, epsilon):
    f64_0 = numba.float64(0)
    n = numba.int32(len(indptr) - 1)
    m = numba.int32(len(indices))
    p = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    alpha_deg = epsilon * np.asarray(deg, dtype=np.float64) / (1. * m)
    r[s] = 1.
    q = np.zeros(n, dtype=np.int32)
    front = numba.int32(0)
    rear = numba.int32(0)
    q_mark = np.zeros(n, dtype=numba.boolean)
    q[rear] = s
    rear += 1
    q_mark[s] = 1
    num_pushes = numba.int32(0)
    while front != rear:
        num_pushes += 1

        uu = q[front % n]
        front += 1
        res = r[uu]
        p[uu] += alpha * res
        rest_prob = ((1. - alpha) * res / deg[uu])
        for vv in indices[indptr[uu]:indptr[uu + 1]]:
            r[vv] += rest_prob
            if r[vv] >= alpha_deg[vv]:
                if not q_mark[vv]:
                    q[rear % n] = vv
                    rear += 1
                    q_mark[vv] = 1
        r[uu] = f64_0
        q_mark[uu] = 0
    return p, r, num_pushes


@jit(nopython=True)
def _calc_ppr_power_iteration(s, indptr, indices, deg, alpha, epsilon, max_iter=500):
    n = numba.int32(len(indptr) - 1)
    p = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    r[s] = 1.
    num_pushes = numba.int32(0)
    power_iter = numba.int32(0)
    for _ in range(max_iter):
        power_iter += 1
        for uu in range(n):
            num_pushes += 1
            p[uu] += alpha * r[uu]
            rest_prob = ((1. - alpha) * r[uu] / deg[uu])
            for vv in indices[indptr[uu]:indptr[uu + 1]]:
                r[vv] += rest_prob
            r[uu] = 0.
        if np.sum(r) < epsilon:
            break
    return p, r, num_pushes, power_iter


@jit(nopython=True)
def _calc_ppr_gradient_descent(s, indptr, indices, alpha, epsilon, max_iter=500):
    n = numba.int32(len(indptr) - 1)
    M = sp.csr_matrix((np.ones(len(indices)), indices, indptr), (n, n))
    S = np.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.spdiags(S.T, 0, *M.shape, format="csr")
    M = Q * M
    p = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    x[s] = alpha
    power_iter = numba.int32(0)
    for _ in range(max_iter):
        x = (1. - alpha) * (x * M) + alpha * p
        # check convergence, l1 norm
        if (1. - np.sum(x)) < epsilon:
            break
    return p, power_iter


def pagerank_nesterov_accer(adj_matrix, alpha, max_iter, p, optimal, tol, beta, init_opt="zero"):
    n, _ = adj_matrix.shape
    M = adj_matrix
    S = np.array(M.sum(axis=1)).flatten()
    if init_opt == 'zero':
        init = np.zeros(n, dtype=np.float64)
    else:
        init = S / np.sum(S)
    x_pre = np.asarray(init, dtype=np.float64)
    x = np.asarray(init, dtype=np.float64)
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.spdiags(S.T, 0, *M.shape, format="csr")
    M = Q * M
    assert p.sum() == 1.
    # power iteration: make up to max_iter iterations
    results = []
    epochs = 0
    for _ in range(max_iter):
        epochs = _ + 1
        yt = x + beta * (x - x_pre)
        x_next = alpha * (yt * M) + (1. - alpha) * p
        x_pre, x = x, x_next
        results.append(np.sum(np.abs(x - optimal)))
        if results[-1] < tol:
            break
    return results, epochs


def ppr_forward_push(csr_mat: sp.csr_matrix, s: int, alpha: np.float64, max_iter: int, epsilon: np.float64,
                     debug=False):
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    n, _ = csr_mat.shape
    m = csr_mat.getnnz()
    p = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    deg = np.array(csr_mat.sum(axis=1)).flatten()
    alpha_deg = epsilon * np.asarray(deg, dtype=np.float64)
    r[s] = 1.
    q = np.zeros(n, dtype=np.int32)
    front = 0
    rear = 0
    q_mark = np.zeros(n, dtype=bool)
    q[rear] = s
    rear += 1
    q_mark[s] = 1
    num_pushes = 0
    results = []
    num_iter = 0
    while front != rear:
        if debug and num_pushes % n == 0:
            results.append(np.copy(p))
            num_iter += 1
            if num_iter >= max_iter:
                break
        uu = q[front % n]
        front += 1
        res = r[uu]
        p[uu] += alpha * res
        if deg[uu] <= 0.:
            continue
        rest_prob = (1. - alpha) * res / deg[uu]
        r[uu] = 0.
        q_mark[uu] = 0
        for vv in indices[indptr[uu]:indptr[uu + 1]]:
            r[vv] += rest_prob
            if r[vv] >= alpha_deg[vv]:
                if not q_mark[vv]:
                    q[rear % n] = vv
                    rear += 1
                    q_mark[vv] = 1
        num_pushes += 1
    return p, r, num_pushes, results


def ppr_fast_push(csr_mat: sp.csr_matrix, s: int, alpha: np.float64, max_iter: int, epsilon: np.float64, debug=False):
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    n, _ = csr_mat.shape
    m = csr_mat.getnnz()
    p = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    deg = np.array(csr_mat.sum(axis=1)).flatten()
    alpha_deg = epsilon * np.asarray(deg, dtype=np.float64) / (1. * m)
    r[s] = 1.
    r_sum = 1.0
    threshold_to_reject = 1. / m
    num_pushes = 0
    previous_num_pushes = 0

    q = np.zeros(n, dtype=np.int32)
    front = 0
    rear = 0
    q_active = np.zeros(n, dtype=bool)
    q[rear] = s
    rear += 1
    q_active[s] = 1
    num_pushes = 0
    results = []
    num_iter = 0
    switch_size = n / 4
    while front != rear and (rear - front) <= switch_size:
        if debug and num_pushes % n == 0:
            results.append(np.copy(p))
            num_iter += 1
            if num_iter >= max_iter:
                break
        uu = q[front % n]
        front += 1
        res = r[uu]
        p[uu] += alpha * res
        if deg[uu] <= 0.:
            continue
        rest_prob = (1. - alpha) * res / deg[uu]
        r[uu] = 0.  # to avoid the self-loop trap.
        q_active[uu] = 0
        for vv in indices[indptr[uu]:indptr[uu + 1]]:
            r[vv] += rest_prob
            if r[vv] >= alpha_deg[vv]:
                if not q_active[vv]:
                    q[rear % n] = vv
                    rear += 1
                    q_active[vv] = 1
        num_pushes += 1
    num_epochs = 0

    return p, r, num_pushes, results


def ppr_forward_push_simultaneous(csr_mat: sp.csr_matrix, s: int, alpha: np.float64, max_iter: int, debug=False):
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    n, _ = csr_mat.shape
    p = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    deg = np.array(csr_mat.sum(axis=1)).flatten()
    r[s] = 1.
    num_pushes = 0
    results = []
    num_iter = 0
    while num_iter < max_iter:
        if debug:
            results.append(np.copy(p))
            num_iter += 1
            if num_iter >= max_iter:
                break
        for uu in range(n):
            if r[uu] <= 0. or deg[uu] <= 0.:
                continue
            res = r[uu]
            p[uu] += alpha * res
            rest_prob = (1. - alpha) * res / deg[uu]
            r[uu] = 0.  # put it first to avoid the self-loop trap.
            num_pushes += 1
            for vv in indices[indptr[uu]:indptr[uu + 1]]:
                r[vv] += rest_prob
    return p, r, num_pushes, results


def ppr_forward_push_asynchronous(csr_mat: sp.csr_matrix, s: int, alpha: np.float64, max_iter: int, debug=False):
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    n, _ = csr_mat.shape
    p = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    deg = np.array(csr_mat.sum(axis=1)).flatten()
    r[s] = 1.
    num_pushes = 0
    results = []
    num_iter = 0
    new_r = np.zeros_like(r)
    while num_iter < max_iter:
        if debug:
            results.append(np.copy(p))
            num_iter += 1
            if num_iter >= max_iter:
                break
        for uu in range(n):
            if r[uu] <= 0. or deg[uu] <= 0.:
                continue
            res = r[uu]
            p[uu] += alpha * res
            rest_prob = (1. - alpha) * res / deg[uu]
            r[uu] = 0.  # put it first to avoid the self-loop trap.
            num_pushes += 1
            for vv in indices[indptr[uu]:indptr[uu + 1]]:
                new_r[vv] += rest_prob

        r, new_r = new_r, r
    return p, r, num_pushes, results


def ppr_power_iteration(csr_mat: sp.csr_matrix, s: int, alpha: np.float64, max_iter: int, debug=False):
    n, _ = csr_mat.shape
    deg = np.array(csr_mat.sum(axis=1)).flatten()
    x = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    v[s] = 1.
    deg[deg != 0] = 1.0 / deg[deg != 0]
    Q = sp.spdiags(deg.T, diags=0, m=n, n=n, format="csr")
    M = Q * csr_mat
    epochs = 0
    results = []
    for _ in range(max_iter):
        if debug:
            results.append(x)
        x = (1. - alpha) * (x * M) + alpha * v
        epochs = _ + 1

    return x, epochs, results


def ppr_coordinate_descent(csr_mat: sp.csr_matrix, s: int, alpha: np.float64, max_iter: int, debug=False):
    n, _ = csr_mat.shape
    deg = np.array(csr_mat.sum(axis=1)).flatten()
    x = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    v[s] = 1.
    deg[deg != 0] = 1.0 / deg[deg != 0]
    Q = sp.spdiags(deg.T, diags=0, m=n, n=n, format="csr")
    M = Q * csr_mat
    epochs = 0
    results = []
    for _ in range(max_iter):
        if debug:
            results.append(x)
        x = (1. - alpha) * (x * M) + alpha * v
        epochs = _ + 1

    return x, epochs, results


def ppr_heavy_ball(csr_mat: sp.csr_matrix, s: int, alpha: np.float64, beta: np.float64, max_iter: int, x0=None,
                   debug=False):
    n, _ = csr_mat.shape
    deg = np.array(csr_mat.sum(axis=1)).flatten()
    v = np.zeros(n, dtype=np.float64)
    v[s] = 1.
    x_pre = np.zeros(n, dtype=np.float64)
    if x0 is None:
        x = np.ones(n, dtype=np.longdouble) / n
    else:
        x = x0
    deg[deg != 0] = 1.0 / deg[deg != 0]
    Q = sp.spdiags(deg.T, diags=0, m=n, n=n, format="csr")
    M = Q * csr_mat
    epochs = 0
    results = []
    beta = (1. - alpha) ** 2. / 2.
    for _ in range(max_iter):
        if debug:
            results.append(x)
        x_next = (1. - alpha) * (x * M) + alpha * v + beta * (x - x_pre)
        epochs = _ + 1
        x_pre, x = x, x_next

    return x, epochs, results


def ppr_ground_truth(csr_mat: sp.csr_matrix, s: int, alpha: np.float64):
    n, _ = csr_mat.shape
    deg = np.array(csr_mat.sum(axis=1)).flatten()
    deg[deg != 0] = 1.0 / deg[deg != 0]
    Q = sp.spdiags(deg.T, diags=0, m=n, n=n, format="csr")
    M = Q * csr_mat
    inv_p = np.linalg.inv(np.identity(n) - (1. - alpha) * M.toarray().T)
    x = np.zeros(n, dtype=np.float64)
    x[s] = 1.
    return alpha * inv_p @ x


@njit(cache=True)
def _forward_push_fifo(indptr, indices, degree, s, eps, alpha):
    n = degree.shape[0]
    queue = np.zeros(n, dtype=np.int32)
    front, rear = np.int32(0), np.int32(1)
    gpr_vec, r = np.zeros(n), np.zeros(n)
    queue[rear] = s
    r[s] = 1.
    q_mark = np.zeros(n)
    q_mark[s] = 1
    eps_vec = np.zeros(n)
    for i in prange(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 0.
    sum_r = 1.
    while front != rear:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        num_oper += degree[u]
        res_u = r[u]
        sum_r -= alpha * res_u
        gpr_vec[u] += alpha * res_u
        rest_prob = (1. - alpha) / degree[u]
        r[u] = 0
        push_amount = rest_prob * res_u
        for v in indices[indptr[u]:indptr[u + 1]]:
            r[v] += push_amount
            if np.abs(r[v]) > eps_vec[u] and not q_mark[v]:
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True
    vec_num_op = np.zeros(2 * n + 2)
    vec_num_op[:n] = gpr_vec
    vec_num_op[n:2 * n] = r
    vec_num_op[-2] = num_oper
    return vec_num_op


@njit(cache=True)
def _forward_push_greedy(indptr, indices, degree, s, eps, alpha):
    n = degree.shape[0]
    gpr_vec, r = np.zeros(n), np.zeros(n)
    r[s] = 1.
    eps_vec = np.zeros(n)
    for i in prange(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 0.
    sum_r = 1.
    while True:
        u = np.argmax(r)
        print(u)
        if np.abs(r[u]) >= eps_vec[u]:
            num_oper += degree[u]
            print(u, (degree[u] * sum_r) / r[u])
            res_u = r[u]
            sum_r -= alpha * res_u
            gpr_vec[u] += alpha * res_u
            rest_prob = (1. - alpha) / degree[u]
            r[u] = 0
            push_amount = rest_prob * res_u
            for v in indices[indptr[u]:indptr[u + 1]]:
                r[v] += push_amount
        else:
            break
    vec_num_op = np.zeros(2 * n + 2)
    vec_num_op[:n] = gpr_vec
    vec_num_op[n:2 * n] = r
    vec_num_op[-2] = num_oper
    return vec_num_op


def forward_push(adj_matrix, s, eps, alpha, n):
    degree = np.int32(adj_matrix.sum(1).A.flatten())
    indices = adj_matrix.indices
    indptr = adj_matrix.indptr
    vec_num_op = _forward_push(indptr, indices, degree, s, eps, alpha)
    gpr_vec = vec_num_op[:n]
    r = vec_num_op[n:2 * n]
    num_oper = vec_num_op[-1]
    return gpr_vec, r, num_oper


def forward_push_fifo(adj_matrix, s, eps, alpha, n):
    degree = np.int32(adj_matrix.sum(1).A.flatten())
    indices = adj_matrix.indices
    indptr = adj_matrix.indptr
    vec_num_op = _forward_push_fifo(indptr, indices, degree, s, eps, alpha)
    gpr_vec = vec_num_op[:n]
    r = vec_num_op[n:2 * n]
    num_oper = vec_num_op[-2]
    max_coef = vec_num_op[-1]
    d_bar = np.mean(degree)
    return gpr_vec, r, num_oper, max_coef, d_bar


def forward_push_greedy(adj_matrix, s, eps, alpha, n):
    degree = np.int32(adj_matrix.sum(1).A.flatten())
    indices = adj_matrix.indices
    indptr = adj_matrix.indptr
    vec_num_op = _forward_push_greedy(indptr, indices, degree, s, eps, alpha)
    gpr_vec = vec_num_op[:n]
    r = vec_num_op[n:2 * n]
    num_oper = vec_num_op[-2]
    max_coef = vec_num_op[-1]
    d_bar = np.mean(degree)
    return gpr_vec, r, num_oper, max_coef, d_bar


def forward_push_mean_v1(indptr, indices, degree, alpha, eps, s, factor):
    start_time = time.time()
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int32)
    front, rear = np.int32(0), np.int32(2)
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = s
    queue[rear] = n  # a funny flag to denote an epoch
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    eps_vec = np.zeros(n + 1)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 0.
    sum_r = 1.
    total_iter = 0
    count_r_nonzero = []
    count_active_nodes = []
    count_num_operations = []
    count_r_sum = []
    count_r_sum_power_iter = []
    active_nodes = 0
    count_num_operations_epoch = []
    num_oper_epoch = 0
    epoch_num = 0
    r_mean = 1.
    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if np.abs(r[u]) > eps_vec[u] and (np.abs(r[u]) >= r_mean):
            active_nodes += 1
            num_oper_epoch += degree[u]
            num_oper += degree[u]
            res_u = r[u]
            sum_r -= alpha * res_u
            gpr_vec[u] += alpha * res_u
            rest_prob = (1. - alpha) / degree[u]
            r[u] = 0
            push_amount = rest_prob * res_u
            for v in indices[indptr[u]:indptr[u + 1]]:
                r[v] += push_amount
                if not q_mark[v]:
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True

        total_iter += 1
        if u == n:
            r_mean = factor * sum_r / np.count_nonzero(r)
            epoch_num += 1
            count_r_sum.append(sum_r)
            count_r_sum_power_iter.append((1 - alpha) ** epoch_num)
            rear = (rear + 1) % n
            queue[rear] = n
            count_r_nonzero.append(np.count_nonzero(r))
            count_active_nodes.append(active_nodes)
            count_num_operations.append(num_oper)
            count_num_operations_epoch.append(num_oper_epoch)
            active_nodes = 0
            num_oper_epoch = 0

    count_r_sum_power_iter = [_ for _ in count_r_sum_power_iter if _ > sum_r]
    results = {"count_r_nonzero": count_r_nonzero,
               "count_active_nodes": count_active_nodes,
               "count_num_operations": count_num_operations,
               "count_r_sum_power_iter": count_r_sum_power_iter,
               "count_num_operations_epoch": count_num_operations_epoch}
    print(f"finish in {time.time() - start_time} with operations: {num_oper}")
    return results


def forward_push_mean_v2(indptr, indices, degree, alpha, eps, s, factor):
    start_time = time.time()
    n = len(degree)
    queue = np.zeros(n + 1, dtype=np.int32)
    front, rear = np.int32(0), np.int32(2)
    gpr_vec, r = np.zeros(n), np.zeros(n + 1)
    queue[1] = s
    queue[rear] = n  # a funny flag to denote an epoch
    r[s] = 1.
    q_mark = np.zeros(n + 1)
    q_mark[s] = 1
    eps_vec = np.zeros(n + 1)
    for i in range(n):
        eps_vec[i] = degree[i] * eps
    num_oper = 0.
    sum_r = 1.
    total_iter = 0
    count_r_nonzero = []
    count_active_nodes = []
    count_num_operations = []
    count_r_sum = []
    count_r_sum_power_iter = []
    active_nodes = 0
    count_num_operations_epoch = []
    num_oper_epoch = 0
    epoch_num = 0
    r_mean = 1.
    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if np.abs(r[u]) > eps_vec[u] and (np.abs(r[u]) >= r_mean):
            active_nodes += 1
            num_oper_epoch += degree[u]
            num_oper += degree[u]
            res_u = r[u]
            sum_r -= alpha * res_u
            gpr_vec[u] += alpha * res_u
            rest_prob = (1. - alpha) / degree[u]
            r[u] = 0
            push_amount = rest_prob * res_u
            for v in indices[indptr[u]:indptr[u + 1]]:
                r[v] += push_amount
                if not q_mark[v]:
                    rear = (rear + 1) % n
                    queue[rear] = v
                    q_mark[v] = True

        total_iter += 1
        if u == n:
            r_mean = factor * sum_r / np.count_nonzero(r)
            epoch_num += 1
            count_r_sum.append(sum_r)
            count_r_sum_power_iter.append((1 - alpha) ** epoch_num)
            rear = (rear + 1) % n
            queue[rear] = n
            count_r_nonzero.append(np.count_nonzero(r))
            count_active_nodes.append(active_nodes)
            count_num_operations.append(num_oper)
            count_num_operations_epoch.append(num_oper_epoch)
            active_nodes = 0
            num_oper_epoch = 0

    count_r_sum_power_iter = [_ for _ in count_r_sum_power_iter if _ > sum_r]
    results = {"count_r_nonzero": count_r_nonzero,
               "count_active_nodes": count_active_nodes,
               "count_num_operations": count_num_operations,
               "count_r_sum_power_iter": count_r_sum_power_iter,
               "count_num_operations_epoch": count_num_operations_epoch}
    print(f"finish in {time.time() - start_time} with operations: {num_oper}")
    return results
