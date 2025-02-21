#pragma once

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


template<typename T> struct pola1{
    T _one;
    CUDA_HOSTDEV pola1(T one=0): _one(one){}

    CUDA_HOSTDEV pola1<T> & operator=(const pola1<T> & a){
        _one = a._one;
        return *this;
    }

    CUDA_HOSTDEV pola1<T>  operator+(const pola1<T> a) const{
        return pola1<T>(_one + a._one);
    }

    CUDA_HOSTDEV pola1<T>& operator+=(const pola1<T>& a){
        _one += a._one;
        return *this;
    }

    CUDA_HOSTDEV pola1<T> operator*(const pola1<T> a) const{
        return pola1<T>(_one * a._one);
    }

    CUDA_HOSTDEV pola1<T> operator*(const T cst) const{
        return pola1<T>(_one*cst);
    }

    CUDA_HOSTDEV pola1<T> operator*(const complex_struct<T> a) const{
        return pola1<T>(_one*a);
    }

    CUDA_HOSTDEV pola1<T>& operator*=(const pola1<T>& a){
        _one*= a._one;
        return *this;
    }

    CUDA_HOSTDEV pola1<T>& operator*=(const T cst){
        _one *= cst;
        return *this;
    }

    CUDA_HOSTDEV pola1<T>& operator*=(const complex_struct<T> a){
        _one *= a;
        return *this;
    }

    CUDA_HOSTDEV pola1<T>& operator/=(const T cst){
        _one /= cst;
        return *this;
    }
};


template<typename T> struct pola2{
    T _one, _two;
    CUDA_HOSTDEV pola2(T one=0, T two=0): _one(one), _two(two){}

    CUDA_HOSTDEV pola2<T>& operator=(const pola2<T> & a){
        _one = a._one;
        _two = a._two;
        return *this;
    }

    CUDA_HOSTDEV pola2<T> operator+(const pola2<T> a){
        return pola2<T>(_one + a._one, _two + a._two);
    }

    CUDA_HOSTDEV pola2<T>& operator+=(const pola2<T>& a){
        _one += a._one;
        _two += a._two;
        return *this;
    }

    CUDA_HOSTDEV pola2<T> operator*(const T cst) const{
        return pola2<T>(_one*cst, _two*cst);
    }

    // Multiplication
    CUDA_HOSTDEV pola2<T> operator*(const pola1<T> a) const{ 
        return pola2<T>(_one*a._one, _two*a._two); 
    }

    CUDA_HOSTDEV pola2<T>& operator*=(const T cst){
        _one *= cst;
        _two *= cst;
        return *this;
    }

    CUDA_HOSTDEV pola2<T>& operator*=(const pola1<T>& a){
        _one *= a._one;
        _two *= a._one;
        return *this;
    }

    CUDA_HOSTDEV pola2<T>& operator/=(const T cst){
        _one /= cst;
        _two /= cst;
        return *this;
    }
};


template<typename T> struct pola4{
    T _one, _two, _three, _four;
    CUDA_HOSTDEV pola4(T one=0, T two=0, T three=0, T four=0): _one(one), _two(two), _three(three), _four(four){}

    CUDA_HOSTDEV pola4<T>& operator=(const pola4<T>& a){
        _one = a._one;
        _two = a._two;
        _three = a._three;
        _four = a._four;
        return *this;
    }

    
    CUDA_HOSTDEV pola4<T>& operator=(const float a){
        _one = a;
        _two = a;
        _three = a;
        _four = a;
        return *this;
    }
    
    

    CUDA_HOSTDEV pola4<T>& operator+=(const pola4<T>& a){
        _one += a._one;
        _two += a._two;
        _three += a._three;
        _four += a._four;
        return *this;
    }


    CUDA_HOSTDEV pola4<T> operator*(const T cst) const{
        return pola4<T>(_one*cst, _two*cst, _three*cst, _four*cst);
    }

    CUDA_HOSTDEV pola4<T> operator*(const pola1<T> a)const{
        return pola4<T>(_one*a._one, _two*a._one, _three*a._one, _four*a._one);
    }

    CUDA_HOSTDEV pola4<T>& operator*=(const T cst){
        _one *= cst;
        _two *= cst;
        _three *= cst;
        _four *= cst;
        return *this;
    }

    CUDA_HOSTDEV pola4<T>& operator*=(const pola1<T>& a){
        _one *= a._one;
        _two *= a._two;
        _three *= a._three;
        _four *= a._four;
        return *this;
    }

    CUDA_HOSTDEV pola4<T>& operator/=(const T cst){
        _one /= cst;
        _two /= cst;
        _three /= cst;
        _four /= cst;
        return *this;
    }
    CUDA_HOSTDEV pola4<T> operator/(const T cst) const{
        return pola4<T>(_one/cst, _two/cst, _three/cst, _four/cst);
    }



};

