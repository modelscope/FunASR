#ifndef TENSOR_H
#define TENSOR_H

#include "alignedmem.h"

using namespace std;

template <typename T> class Tensor {
  private:
    void alloc_buff();
    void free_buff();
    int mem_size;

  public:
    T *buff;
    int size[4];
    int buff_size;
    Tensor(Tensor<T> *in);
    Tensor(int a);
    Tensor(int a, int b);
    Tensor(int a, int b, int c);
    Tensor(int a, int b, int c, int d);
    ~Tensor();
    void zeros();
    void shape();
    void disp();
    void dump(const char *mode);
    void concat(Tensor<T> *din, int dim);
    void resize(int a, int b, int c, int d);
    void add(float coe, Tensor<T> *in);
    void add(Tensor<T> *in);
    void add(Tensor<T> *in1, Tensor<T> *in2);
    void reload(Tensor<T> *in);
};

template <typename T> Tensor<T>::Tensor(int a) : size{1, 1, 1, a}
{
    alloc_buff();
}

template <typename T> Tensor<T>::Tensor(int a, int b) : size{1, 1, a, b}
{
    alloc_buff();
}

template <typename T> Tensor<T>::Tensor(int a, int b, int c) : size{1, a, b, c}
{

    alloc_buff();
}

template <typename T>
Tensor<T>::Tensor(int a, int b, int c, int d) : size{a, b, c, d}
{
    alloc_buff();
}

template <typename T> Tensor<T>::Tensor(Tensor<T> *in)
{
    memcpy(size, in->size, 4 * sizeof(int));
    alloc_buff();
    memcpy(buff, in->buff, in->buff_size * sizeof(T));
}

template <typename T> Tensor<T>::~Tensor()
{
    free_buff();
}

template <typename T> void Tensor<T>::alloc_buff()
{
    buff_size = size[0] * size[1] * size[2] * size[3];
    mem_size = buff_size;
    buff = (T *)AlignedMalloc(32, buff_size * sizeof(T));
}

template <typename T> void Tensor<T>::free_buff()
{
    aligned_free(buff);
}

template <typename T> void Tensor<T>::zeros()
{
    memset(buff, 0, buff_size * sizeof(T));
}

template <typename T> void Tensor<T>::shape()
{
    printf("(%d,%d,%d,%d)\n", size[0], size[1], size[2], size[3]);
}

// TODO:: fix it!!!!
template <typename T> void Tensor<T>::concat(Tensor<T> *din, int dim)
{
    memcpy(buff + buff_size, din->buff, din->buff_size * sizeof(T));
    buff_size += din->buff_size;
    size[dim] += din->size[dim];
}

// TODO:: fix it!!!!
template <typename T> void Tensor<T>::resize(int a, int b, int c, int d)
{
    size[0] = a;
    size[1] = b;
    size[2] = c;
    size[3] = d;
    buff_size = size[0] * size[1] * size[2] * size[3];
}

template <typename T> void Tensor<T>::add(float coe, Tensor<T> *in)
{
    int i;
    for (i = 0; i < buff_size; i++) {
        buff[i] = buff[i] + coe * in->buff[i];
    }
}

template <typename T> void Tensor<T>::add(Tensor<T> *in)
{
    int i;
    for (i = 0; i < buff_size; i++) {
        buff[i] = buff[i] + in->buff[i];
    }
}

template <typename T> void Tensor<T>::add(Tensor<T> *in1, Tensor<T> *in2)
{
    int i;
    for (i = 0; i < buff_size; i++) {
        buff[i] = buff[i] + in1->buff[i] + in2->buff[i];
    }
}

template <typename T> void Tensor<T>::reload(Tensor<T> *in)
{
    memcpy(buff, in->buff, in->buff_size * sizeof(T));
}

template <typename T> void Tensor<T>::disp()
{
    int i;
    for (i = 0; i < buff_size; i++) {
        cout << buff[i] << " ";
    }
    cout << endl;
}

template <typename T> void Tensor<T>::dump(const char *mode)
{
    FILE *fp;
    fp = fopen("tmp.bin", mode);
    fwrite(buff, 1, buff_size * sizeof(T), fp);
    fclose(fp);
}
#endif
