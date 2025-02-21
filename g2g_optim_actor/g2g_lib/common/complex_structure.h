


#pragma once

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


template<typename T> struct complex_struct{
    T _real, _imag;
    CUDA_HOSTDEV complex_struct(T real=0, T imag=0): _real(real), _imag(imag){}

    CUDA_HOSTDEV complex_struct<T>& operator=(const complex_struct<T>& a){
        _real = a._real;
        _imag = a._imag;
        return *this;
    }
    
    CUDA_HOSTDEV complex_struct<T> operator+(const complex_struct<T>& a)const{
        return complex_struct<T>(_real + a._real, _imag + a._imag);
    }

    CUDA_HOSTDEV complex_struct<T>& operator+=(const complex_struct<T>& a){
        _real = _real + a._real;
        _imag = _imag + a._imag;
        return *this;
    }

    CUDA_HOSTDEV complex_struct<T> operator*(const T cst)const{
        return complex_struct<T>(_real*cst, _imag*cst);
    }


    CUDA_HOSTDEV complex_struct<T> operator*(const complex_struct<T>& a)const{
        T re = _real*a._real - _imag*a._imag;
        T im = _imag*a._real + _real*a._imag;
        return complex_struct<T>(re, im);
    }

    CUDA_HOSTDEV complex_struct<T>& operator*=(const T cst){
        _real *= cst;
        _imag *= cst;
        return *this;
    }
    
    CUDA_HOSTDEV complex_struct<T>& operator*=(const complex_struct<T>& a){
        T re = _real*a._real - _imag*a._imag;
        T im = _imag*a._real + _real*a._imag;
        _real = re;
        _imag = im;
        return *this;
    }

    CUDA_HOSTDEV complex_struct<T>& operator/=(const T cst){
        _real = _real/cst;
        _imag = _imag/cst;
        return *this;
    }

    CUDA_HOSTDEV complex_struct<T> operator/(const T cst)const{
        return complex_struct<T>(_real/cst, _imag/cst);
    }




    /* TODO : conjugate */
};
