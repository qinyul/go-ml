package main

import (
	"fmt"
	"math"
	"math/rand"
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
	return float64(1) / (float64(1) + float64(math.Exp(-x)))
}

func matSum(dst *mat, a mat) {
	if dst.rows == a.rows && dst.cols == a.cols {
		for i := 0; i < dst.rows; i++ {
			for j := 0; j < dst.cols; j++ {
				dst.mat[i][j] = dst.mat[i][j] + a.mat[i][j]
			}
		}
	} else {
		panic("invalid matrix, dst and a is not equal")
	}
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

func (m *mat) generateColumn() {

	for i := 0; i < m.rows; i++ {
		var newCols []float64
		for j := 0; j < m.cols; j++ {
			newCols = append(newCols, float64(1))
		}
		m.mat = append(m.mat, newCols)
	}
}

func (m *mat) matRand(low float64, high float64) {
	for i := 0; i < len(m.mat); i++ {
		for j := 0; j < len(m.mat[i]); j++ {
			m.mat[i][j] = rand.Float64()*(high-low) + low
		}
	}
}

func (m *mat) matFil(x float64) {
	for i := 0; i < len(m.mat); i++ {
		for j := 0; j < len(m.mat[i]); j++ {
			m.mat[i][j] = x
		}
	}
}

func (m *mat) matCopy(src mat) {
	if m.rows != src.rows {
		panic("matCopy:: rows not equal")
	}
	if m.cols != src.cols {
		panic("matCopy:: cols not equal")
	}

	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.mat[i][j] = src.mat[i][j]
		}
	}
}

func (m *mat) matSig() {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.mat[i][j] = sigmoidf(m.mat[i][j])
		}
	}
}

func (m *mat) matPrint() {
	fmt.Printf("%s = [\n", m.name)
	for i := 0; i < len(m.mat); i++ {
		for j := 0; j < len(m.mat[i]); j++ {
			fmt.Printf("    %f ", m.mat[i][j])
		}
		fmt.Printf("\n")
	}
	fmt.Printf("]\n")
}

func getMatrixValue(mat Matrix) float64 {
	lastRow := len(mat) - 1
	lastColumn := len(mat[lastRow]) - 1
	lastValue := mat[lastRow][lastColumn]
	return lastValue
}
