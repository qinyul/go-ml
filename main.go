package main

import (
	"fmt"

	mlMat "github.com/ml/matrix"
	neuralNetwork "github.com/ml/neuralnetwork"
	"github.com/ml/visualization"
)

var (
	td = mlMat.Matrix{
		[]float64{0, 0, 0},
		[]float64{0, 1, 1},
		[]float64{1, 0, 1},
		[]float64{1, 1, 0},
	}
	tdx = mlMat.Matrix{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1},
	}
	tdy = mlMat.Matrix{
		[]float64{0},
		[]float64{1},
		[]float64{1},
		[]float64{0},
	}
	arch = neuralNetwork.Arc{
		Spec: []int{2, 2, 1},
		WS:   []neuralNetwork.ArcDetail{},
		BS:   []neuralNetwork.ArcDetail{},
		AS:   []neuralNetwork.ArcDetail{},
	}
)

func main() {

	stride := 3
	n := 4
	// eps := 1e-1
	rate := 1

	ti := mlMat.Mat{
		Rows:   n,
		Cols:   2,
		Name:   "ti",
		Stride: stride,
		Mat:    tdx,
	}
	to := mlMat.Mat{
		Rows:   n,
		Cols:   1,
		Name:   "to",
		Stride: stride,
		Mat:    tdy,
	}

	nn := neuralNetwork.NNAlloc(&arch, len(arch.Spec), true)

	g := neuralNetwork.NNAlloc(&arch, len(arch.Spec), false)

	neuralNetwork.NNPrint(nn, "nn")
	vs := visualization.Visualization{
		NN:   &nn,
		Arch: arch,
		G:    &g,
		TI:   ti,
		TO:   to,
	}

	// switch to use visualization
	if true {
		vs.InitVisualization()

		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				nn.AS[0].Mat[0][0] = float64(i)
				nn.AS[0].Mat[0][1] = float64(j)
				neuralNetwork.NNForward(nn)
				fmt.Printf("%d ^ %d = %f\n", i, j, nn.AS[len(nn.AS)-1].Mat[0][0])
			}
		}
	} else {
		fmt.Printf("cost = %f\n", neuralNetwork.NNCost(nn, ti, to))

		for i := 0; i < 100*1000; i++ {
			neuralNetwork.NNBackprop(nn, g, ti, to)
			neuralNetwork.NNLearn(nn, g, float64(rate))

			fmt.Printf("%d: cost = %f\n", i, neuralNetwork.NNCost(nn, ti, to))

		}

		// NNPrint(nn, "nn")
		//
		fmt.Printf("---TRUTH TABLE----\n")
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				nn.AS[0].Mat[0][0] = float64(i)
				nn.AS[0].Mat[0][1] = float64(j)
				neuralNetwork.NNForward(nn)
				fmt.Printf("%d ^ %d = %f\n", i, j, nn.AS[len(nn.AS)-1].Mat[0][0])
			}
		}
	}

}
