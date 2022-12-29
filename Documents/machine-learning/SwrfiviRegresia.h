#ifndef ML_LINEARREGRESSION_H
#define ML_LINEARREGRESSION_H

// თემო ლომსაძე - მანქანური სწავლების პროექტი - წრფივი რეგრესიის კოდური იმპლემენტაცია

class LinearRegression {

public:
    double *x;
    double *y;
    int m;
    double *theta;
    LinearRegression(double x[], double y[], int m);
    /**
     * @param alpha
     * @param iterations
     */

    void train(double alpha, int iterations);
    double predict(double x);

private:
    static double compute_cost(double x[], double y[], double theta[], int m);
    static double h(double x, double theta[]);
    static double *calculate_predictions(double x[], double theta[], int m);
    static double *gradient_descent(double x[], double y[], double alpha, int iters, double *J, int m);
};


#endif