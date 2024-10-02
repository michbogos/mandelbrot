#include<immintrin.h>

__m256i MandAVX (__m256d initx, __m256d inity, __m256i iterations) {
    __m256d cx,cy,xsq,ysq, four, mask1, two;
    __m256i iter, mask2, one;
    four = _mm256_set1_pd(4.0);
    one = _mm256_set1_epi64x(1);
    iter = _mm256_set1_epi64x(0);
    two = _mm256_set1_pd(2.0);

    // cx=initx+initx*initx - inity*inity;   
    // cy=inity+initx*inity+initx*inity; 
    cx = _mm256_sub_pd(_mm256_fmadd_pd(initx, initx, initx), _mm256_mul_pd(inity, inity));
    cy = _mm256_add_pd(inity, _mm256_mul_pd(two, _mm256_mul_pd(initx, inity)));

    ysq = _mm256_mul_pd(cy, cy);
    xsq = _mm256_mul_pd(cx, cx);

    repeat:
        cy = _mm256_add_pd(inity, _mm256_mul_pd(two, _mm256_mul_pd(initx, inity)));
        cx = _mm256_sub_pd(initx, _mm256_add_pd(ysq, xsq));

        ysq = _mm256_mul_pd(cy, cy);
        xsq = _mm256_mul_pd(cx, cx);

        mask1 = _mm256_cmp_pd(_mm256_add_pd(ysq, xsq), four, _CMP_LT_OQ);
        mask2 = _mm256_cmpgt_epi64(iterations, iter);

        mask2 = _mm256_and_si256(mask2, _mm256_castpd_si256(mask1));

        iter = _mm256_add_epi64(iter, one);

        if(_mm256_movemask_pd(mask2)>0){
            goto repeat;
        }
    
    // for (iter=0;iter<iterations && (ysq=cy*cy)+(xsq=cx*cx)<4;iter++,cy=inity+cx*cy+cx*cy,cx=initx-ysq+xsq) ;
    // return (iter);
    return iter;
}

unsigned int Mand (double initx, double inity, unsigned int iterations) {
    
    double cx,cy,xsq,ysq;
    unsigned int iter;
    
    cx=initx+initx*initx - inity*inity;   
    cy=inity+initx*inity+initx*inity; 
    
    for (iter=0;iter<iterations && (ysq=cy*cy)+(xsq=cx*cx)<4;iter++,cy=inity+cx*cy+cx*cy,cx=initx-ysq+xsq) ;
    return (iter);
}

int main(){
    return 0;
}