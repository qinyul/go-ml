package main

import (
	"fmt"
)

var (
	td = Matrix{
		[]float64{0, 0, 0},
		[]float64{0, 1, 1},
		[]float64{1, 0, 1},
		[]float64{1, 1, 0},
	}
	tdx = Matrix{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1},
	}
	tdy = Matrix{
		[]float64{0},
		[]float64{1},
		[]float64{1},
		[]float64{0},
	}
	arch = arc{
		spec: []int{2, 4, 1},
		ws:   []arcDetail{},
		bs:   []arcDetail{},
		as:   []arcDetail{},
	}
)

func main() {

	stride := 3
	n := 4
	eps := 1e-1
	rate := 1e-1

	ti := mat{
		rows:   n,
		cols:   2,
		name:   "ti",
		stride: stride,
		mat:    tdx,
	}
	to := mat{
		rows:   n,
		cols:   1,
		name:   "to",
		stride: stride,
		mat:    tdy,
	}

	nn := nnAlloc(&arch, len(arch.spec), true)

	g := nnAlloc(&arch, len(arch.spec), false)

	fmt.Printf("cost = %f\n", nnCost(nn, ti, to))

	for i := 0; i < 20*1000; i++ {
		nnFiniteDiff(nn, g, eps, ti, to)
		nnLearn(nn, g, rate)

		fmt.Printf("%d: cost = %f\n", i, nnCost(nn, ti, to))
	}

	NNPrint(nn, "nn")

	fmt.Printf("---TRUTH TABLE----\n")
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			nn.as[0].mat[0][0] = float64(i)
			nn.as[0].mat[0][1] = float64(j)
			nnForward(nn)
			fmt.Printf("%d ^ %d = %f\n", i, j, nn.as[len(nn.as)-1].mat[0][0])
		}
	}

}
