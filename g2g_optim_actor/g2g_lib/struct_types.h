

#pragma once


struct uvw_coord_t{
    float _u, _v, _w;
    uvw_coord_t(float u=0, float v=0, float w=0): _u(u), _v(v), _w(w){}


    uvw_coord_t& operator=(const uvw_c& a){
        _u = a._u;
        _v = a._v;
        _w = a._w;
        return *this;
    }

    /* Scalar multiplication */
    uvw_coord_t& operator*=(const float scal){
        _u *= scal;
        _v *= scal;
        _w *= scal;
        return *this;
    }

    /* element multiplication */
    uvw_coord_t& operator*=(const uvw_c& a){
        _u *= a._u;
        _v *= a._v;
        _w *= a._w;
        return *this;
    }
};



struct complex_t{
    float _real, _imag;
    complex_t(float real=0, float imag=0): _real(real), _imag(imag){}

    complex_t& operator=(const complex_t& a){
        _real = a._real;
        _imag = a._imag;
        return *this;
    }

    bool& operator==(const complex_t& a){
        return ( (_real == a._real) && (_imag == a._imag));
    }

    complex_t& operator+(const complex_t& a){
        _real = _real + a._real;
        _imag = _imag + a._imag;
        return *this;
    }

    complex_t& operator+=(const complex_t& a){
        _real += a._real;
        _imag += a._imag;
        return *this;
    }

    complex_t& operator *(const complex_t& a){
        float new_real = _real*a._real - _imag*a._imag;
        float new_imag = _real*a._imag + _imag*a._real;
        _real = new_real;
        _imag = new_imag;
        return *this;
    }

    complex_t& operator *(const float scal){
        _real = _real*scal;
        _imag = _imag*scal;
        return *this;
    }

    /* TODO : conjugate */
}

