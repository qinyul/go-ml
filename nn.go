package main

import (
	"fmt"
	"math/rand/v2"
)

type NN struct {
	count int
	ws    []mat
	bs    []mat
	as    []mat
}

type arcDetail struct {
}

type arc struct {
	spec []int
	ws   []arcDetail
	bs   []arcDetail
	as   []arcDetail
}

func nnAlloc(arch *arc, archCount int, useRandValue bool) NN {
	var nn NN
	nn.count = archCount - 1

	nn.as = append(nn.as, mat{
		name: "as0",
		rows: 1,
		cols: arch.spec[0],
		mat:  generateNNColumn(1, arch.spec[0], false),
	})

	for i := 1; i < archCount; i++ {
		nn.ws = append(nn.ws, mat{
			rows: nn.as[i-1].cols,
			name: fmt.Sprintf("ws%d", i-1),
			cols: arch.spec[i],
			mat:  generateNNColumn(nn.as[i-1].cols, arch.spec[i], useRandValue),
		})
		nn.bs = append(nn.bs, mat{
			rows: 1,
			name: fmt.Sprintf("bs%d", i-1),
			cols: arch.spec[i],
			mat:  generateNNColumn(1, arch.spec[i], useRandValue),
		})
		nn.as = append(nn.as, mat{
			rows: 1,
			name: fmt.Sprintf("as%d", i),
			cols: arch.spec[i],
			mat:  generateNNColumn(1, arch.spec[i], false),
		})

	}
	return nn
}

func NNPrint(nn NN, name string) {
	fmt.Printf("%s = [\n", name)
	for i := 0; i < nn.count; i++ {
		nn.ws[i].matPrint(2)
		nn.bs[i].matPrint(2)
	}
	fmt.Printf("]\n")
}

func generateNNColumn(rows int, cols int, useRandValue bool) Matrix {
	value := float64(0)

	var result Matrix
	for i := 0; i < rows; i++ {
		var newCols []float64
		for j := 0; j < cols; j++ {
			if useRandValue {
				value = rand.Float64()
			}
			newCols = append(newCols, value)
		}
		result = append(result, newCols)
	}

	return result
}

func nnForward(nn NN) {
	for i := 0; i < nn.count; i++ {
		matDot(&nn.as[i+1], nn.as[i], nn.ws[i])
		matSum(&nn.as[i+1], nn.bs[i])
		nn.as[i+1].matSig()
	}

}

func nnCost(nn NN, ti mat, to mat) float64 {

	nnOutput := nn.as[len(nn.as)-1]

	if ti.rows != to.rows {
		panic("NNcost:: rows not equal")
	}
	if to.cols != nnOutput.cols {
		panic("NNcost:: to cols and nnOutput.cols not equal")
	}

	n := ti.rows

	c := float64(0)

	for i := 0; i < n; i++ {

		x := mat{
			rows: 1,
			name: fmt.Sprintf("ti-%d", i),
			cols: ti.cols,
			mat: Matrix{
				ti.mat[i],
			},
		}

		y := mat{
			mat: Matrix{
				to.mat[i],
			},
		}

		nn.as[0].matCopy(x)
		nnForward(nn)

		q := to.cols
		for j := 0; j < q; j++ {
			d := nnOutput.mat[0][j] - y.mat[0][j]
			c += d * d
		}
	}

	return c / float64(n)
}

func nnFiniteDiff(nn NN, g NN, eps float64, ti mat, to mat) {
	var saved float64
	c := nnCost(nn, ti, to)

	for i := 0; i < nn.count; i++ {
		for j := 0; j < nn.ws[i].rows; j++ {
			for k := 0; k < nn.ws[i].cols; k++ {
				saved = nn.ws[i].mat[j][k]
				nn.ws[i].mat[j][k] += eps
				g.ws[i].mat[j][k] = (nnCost(nn, ti, to) - c) / eps
				nn.ws[i].mat[j][k] = saved
			}
		}

		for j := 0; j < nn.bs[i].rows; j++ {
			for k := 0; k < nn.bs[i].cols; k++ {
				saved = nn.bs[i].mat[j][k]
				nn.bs[i].mat[j][k] += eps
				g.bs[i].mat[j][k] = (nnCost(nn, ti, to) - c) / eps
				nn.bs[i].mat[j][k] = saved
			}
		}
	}
}

func nnLearn(nn NN, g NN, rate float64) {
	for i := 0; i < nn.count; i++ {
		for j := 0; j < nn.ws[i].rows; j++ {
			for k := 0; k < nn.ws[i].cols; k++ {
				nn.ws[i].mat[j][k] -= rate * g.ws[i].mat[j][k]
			}
		}

		for j := 0; j < nn.bs[i].rows; j++ {
			for k := 0; k < nn.bs[i].cols; k++ {
				nn.bs[i].mat[j][k] -= rate * g.bs[i].mat[j][k]
			}
		}
	}
}
