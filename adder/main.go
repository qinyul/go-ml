package main

import (
	"fmt"

	mlMat "github.com/ml/matrix"
	neuralNetwork "github.com/ml/neuralnetwork"
)

const (
	BITS = 3
)

func main() {
	n := (1 << BITS)
	rows := n * n
	ti := mlMat.Mat{
		Rows: rows,
		Cols: 2 * BITS,
		Name: "ti",
		Mat:  neuralNetwork.GenerateNNColumn(rows, 2*BITS, false),
	}

	to := mlMat.Mat{
		Rows: rows,
		Cols: BITS + 1,
		Name: "to",
		Mat:  neuralNetwork.GenerateNNColumn(rows, BITS+1, false),
	}

	var ol float64
	for i := 0; i < ti.Rows; i++ {
		x := i / n
		y := i % n
		z := x + y
		overflow := z >= n
		if overflow {
			ol = 1
		} else {
			ol = 0
		}
		for j := 0; j < BITS; j++ {
			ti.Mat[i][j] = float64(x >> j & 1)
			ti.Mat[i][j+BITS] = float64(y >> j & 1)

			if overflow {
				to.Mat[i][j] = 0
			} else {
				to.Mat[i][j] = float64(z >> j & 1)
			}
		}

		to.Mat[i][BITS] = float64(ol)
	}
	arch := neuralNetwork.Arc{
		Spec: []int{2 * BITS, 2 * BITS, BITS + 1},
		WS:   []neuralNetwork.ArcDetail{},
		BS:   []neuralNetwork.ArcDetail{},
		AS:   []neuralNetwork.ArcDetail{},
	}
	nn := neuralNetwork.NNAlloc(&arch, len(arch.Spec), true)
	g := neuralNetwork.NNAlloc(&arch, len(arch.Spec), true)
	neuralNetwork.NNPrint(nn, "nn")
	fmt.Printf("c = %f\n", neuralNetwork.NNCost(nn, ti, to))

	rate := float64(1)

	// i = learning attempt
	for i := 0; i < 10*1; i++ {
		// to swith between back propagation and finite different method
		if true {
			neuralNetwork.NNBackprop(nn, g, ti, to)
		} else {
			neuralNetwork.NNFiniteDiff(nn, g, 1e-1, ti, to)
		}

		neuralNetwork.NNLearn(nn, g, rate)
		fmt.Printf("%d: c = %f\n", i, neuralNetwork.NNCost(nn, ti, to))
	}

	fails := 0
	for x := 0; x < n; x++ {
		for y := 0; y < n; y++ {

			z := x + y
			for j := 0; j < BITS; j++ {
				nn.AS[0].Mat[0][j] = float64((x >> j) & 1)
				nn.AS[0].Mat[0][j+BITS] = float64((y >> j) & 1)
			}

			neuralNetwork.NNForward(nn)
			if (nn.AS[len(nn.AS)-1]).Mat[0][BITS] > float64(0.5) {
				if z < n {
					fmt.Printf("%d + %d = (OVERFLOW<>%d)\n", x, y, z)
					fails += 1
				}
			} else {
				a := 0
				for j := 0; j < BITS; j++ {
					bit := 0
					if nn.AS[len(nn.AS)-1].Mat[0][j] > float64(0.5) {
						bit = 1
					}
					a |= bit << j
				}
				if z != a {
					fmt.Printf("%d + %d = (expected: %d <> actual:%d)\n", x, y, z, a)
					fails += 1
				}
			}
		}
	}

	if fails == 0 {
		fmt.Printf("Total Failure = %d, Status: OK\n", fails)
	} else {
		fmt.Printf("Total Failure = %d, Status: Failed\n", fails)
	}
}
