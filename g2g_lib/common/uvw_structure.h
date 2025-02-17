#pragma once

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


template<typename T> struct uvw_struct{
    T _u, _v, _w;
    CUDA_HOSTDEV uvw_struct(T u=0, T v=0, T w=0): _u(u), _v(v), _w(w){}

    CUDA_HOSTDEV uvw_struct<T>& __restrict__ operator*(const T cst) const{
        return uvw_struct(_u*cst, _v*cst, _w*cst);
    }

    CUDA_HOSTDEV uvw_struct<T>& __restrict__ operator*=(const T cst){
        _u *= cst;
        _v *= cst;
        _w *= cst;
        return *this;
    }

    CUDA_HOSTDEV uvw_struct<T>& __restrict__ operator/(const T cst) const{
        return uvw_struct(_u/cst, _v/cst, _w/cst);
    }

    CUDA_HOSTDEV uvw_struct<T>& __restrict__ operator/=(const T cst){
        _u /= cst;
        _v /= cst;
        _w /= cst;
        return *this;
    }
};
