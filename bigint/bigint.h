#ifndef BIGINT_H
#define BIGINT_H
#endif

typedef struct bigint_t{
    unsigned long* limbs;
    unsigned char  num_limbs;
    bool sign;
};