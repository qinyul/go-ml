package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	mlMat "github.com/ml/matrix"
	neuralNetwork "github.com/ml/neuralnetwork"
	"github.com/ml/visualization"
)

const (
	SCREEN_HEIGHT = 700
	SCREEN_WIDTH  = 700
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

func extractText(arch *neuralNetwork.Arc) {
	file, err := os.Open("./xor.arch")
	if err != nil {
		fmt.Println("Error opening file: ", err)
		panic("error opening file")
	}

	defer file.Close()

	result := ""
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		result = fmt.Sprintf("%s %s", result, scanner.Text())
	}

	var datas []int
	for _, v := range strings.Split(result, " ") {
		if len(v) > 0 {
			d, err := strconv.Atoi(v)
			if err == nil {
				datas = append(datas, d)
			} else {
				fmt.Printf("%s is not a number\n", v)
			}

		}
	}
	fmt.Printf("Input Data: %v\n", datas)

	arch.Spec = datas
}

func main() {
	importMode := flag.Bool("importMode", false, "switch between static data mode or import mode")

	flag.Parse()

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

	if *importMode {
		t := mlMat.LoadMatJSON("./xor-mat.json")
		extractText(&arch)

		inSize := arch.Spec[0]
		outSize := arch.Spec[len(arch.Spec)-1]

		if t.Cols != inSize+outSize {
			fmt.Printf("t.Cols = %d, inSize=%d, outSize=%d\n", t.Cols, inSize, outSize)
			panic("t.Cols not equal with inSize + outSize")
		}

		if len(arch.Spec) < 1 {
			panic("arch spec dont have data")
		}

		ti = mlMat.Mat{
			Rows: t.Rows,
			Name: ti.Name,
			Cols: inSize,
			Mat:  neuralNetwork.GenerateNNColumn(t.Rows, inSize, false),
		}

		to = mlMat.Mat{
			Rows: t.Rows,
			Name: to.Name,
			Cols: outSize,
			Mat:  neuralNetwork.GenerateNNColumn(t.Rows, outSize, false),
		}
	}

	nn := neuralNetwork.NNAlloc(&arch, len(arch.Spec), true)

	g := neuralNetwork.NNAlloc(&arch, len(arch.Spec), false)

	neuralNetwork.NNPrint(nn, "nn")

	rw := SCREEN_WIDTH / 2
	rh := SCREEN_HEIGHT * 2 / 3
	rx := SCREEN_WIDTH - rw
	ry := SCREEN_HEIGHT/2 - rh/2

	vs := visualization.Visualization{
		NN:   &nn,
		Arch: arch,
		G:    &g,
		TI:   ti,
		TO:   to,
		RW:   rw,
		RH:   rh,
		RX:   rx,
		RY:   ry,
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
