#ifndef ML_UTILS_H
#define ML_UTILS_H

// თემო ლომსაძე - მანქანური სწავლების პროექტი - წრფივი რეგრესიის კოდური იმპლემენტაცია

#include <vector>
class Utils {

public:
    static double array_sum(double arr[], int len);
    static double *array_pow(double arr[], int len, int power);
    static double *array_multiplication(double arr1[], double arr2[], int len);
    static double *array_diff(double arr1[], double arr2[], int len);

};

#endif