package main

import (
	"fmt"
	"math"
)

type Matrix [][]float64

type mat struct {
	rows   int
	cols   int
	name   string
	stride int
	mat    Matrix
}

func sigmoidf(x float64) float64 {
	return float64(1) / (float64(1) + math.Exp(-x))
}

func matDot(dst *mat, a mat, b mat) {
	if a.cols != b.rows {
		panic("a.cols and b.rows not equal")
	}

	if dst.rows != a.rows {
		panic("dst.rows and a.rows not equal")
	}

	if dst.cols != b.cols {
		panic("dst.cols and b.cols not equal")
	}

	n := a.cols
	for i := 0; i < dst.rows; i++ {
		for j := 0; j < dst.cols; j++ {
			dst.mat[i][j] = 0
			for k := 0; k < n; k++ {
				dst.mat[i][j] += a.mat[i][k] * b.mat[k][j]
			}
		}
	}
}

func (m *mat) matCopy(src mat) {
	if m.rows != src.rows {
		panic("matCopy:: rows not equal")
	}

	if m.cols != src.cols {
		src.matPrint(0)
		fmt.Printf("m.cols = %d, src.cols = %d\n", m.cols, src.cols)
		panic("matCopy:: cols not equal")
	}

	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.mat[i][j] = src.mat[i][j]
		}
	}
}

func matSum(dst *mat, a mat) {
	if dst.rows == a.rows && dst.cols == a.cols {
		for i := 0; i < dst.rows; i++ {
			for j := 0; j < dst.cols; j++ {
				dst.mat[i][j] += a.mat[i][j]
			}
		}
	} else {
		panic("invalid matrix, dst and a is not equal")
	}
}

func (m *mat) matSig() {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.mat[i][j] = sigmoidf(m.mat[i][j])
		}
	}
}

func (m *mat) matPrint(padding int) {
	fmt.Printf("%*s%s = [\n", padding, "", m.name)
	for i := 0; i < len(m.mat); i++ {
		for j := 0; j < len(m.mat[i]); j++ {
			fmt.Printf("%*s    ", padding, "")
			fmt.Printf("%f ", m.mat[i][j])
		}
		fmt.Printf("\n")
	}
	fmt.Printf("%*s]\n", padding, "")
}
