## SGEMM
SGEMM（Single-Precision General Matrix Multiply）是 BLAS（Basic Linear Algebra Subprograms）库中的一个常用函数，执行单精度矩阵乘法。常被当作矩阵优化测试样例。

在本例中样例为：
C = A × B + C

定义三个矩阵：

$$
A =
\begin{bmatrix}
2 & 8 \\
5 & 1 \\
4 & 2 \\
8 & 6
\end{bmatrix},\quad
B =
\begin{bmatrix}
10 & 9 & 5 \\
5 & 9 & 4
\end{bmatrix},\quad
C =
\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1
\end{bmatrix}
$$


$$
\text{res} = A \times B + C
$$

计算结果为：

$$
\text{res} =
\begin{bmatrix}
61 & 56 & 51 & 111 \\
91 & 55 & 55 & 127 \\
43 & 30 & 29 & 65
\end{bmatrix}
$$


GGML 的内部实现：使用转置优化
在 GGML 中，为了提升内存访问效率和计算性能，矩阵乘法函数 ggml_mul_mat 实际接收的是 B 的转置形式
结果C也是转置的

$$
\text{mulmat}(A, B^T) = C^T
$$

$$
\text{mulmat}\left(
\begin{bmatrix}
2 & 8 \\
5 & 1 \\
4 & 2 \\
8 & 6
\end{bmatrix},
\begin{bmatrix}
10 & 5 \\
9 & 9 \\
5 & 4
\end{bmatrix}
\right)
=
\begin{bmatrix}
60 & 55 & 50 & 110 \\
90 & 54 & 54 & 126 \\
42 & 29 & 28 & 64
\end{bmatrix}
$$