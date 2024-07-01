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
	BITS          = 3
	SCREEN_HEIGHT = 700
	SCREEN_WIDTH  = 700
)

func extractText(arch *neuralNetwork.Arc) {
	file, err := os.Open("../adder.arch")
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

	if *importMode {
		t := mlMat.LoadMatJSON("../mat.json")
		t.MatPrint(0)
		extractText(&arch)
		if len(arch.Spec) < 1 {
			panic("arch spec dont have data")
		}

		inSize := arch.Spec[0]
		outSize := arch.Spec[len(arch.Spec)-1]

		if t.Cols != inSize+outSize {
			fmt.Printf("t.Cols = %d, inSize=%d, outSize=%d\n", t.Cols, inSize, outSize)
			panic("t.Cols not equal with inSize + outSize")
		}

		ti = mlMat.Mat{
			Rows: t.Rows,
			Name: ti.Name,
			Cols: inSize,
			Mat:  neuralNetwork.GenerateNNColumn(t.Rows, inSize, true),
		}

		for i := 0; i < ti.Rows; i++ {
			for j := 0; j < ti.Cols; j++ {
				ti.Mat[i][j] = t.Mat[i][j]
			}
		}

		to = mlMat.Mat{
			Rows: t.Rows,
			Name: to.Name,
			Cols: outSize,
			Mat:  neuralNetwork.GenerateNNColumn(t.Rows, outSize, true),
		}

	}

	nn := neuralNetwork.NNAlloc(&arch, len(arch.Spec), true)
	g := neuralNetwork.NNAlloc(&arch, len(arch.Spec), true)
	neuralNetwork.NNPrint(nn, "nn")

	fmt.Printf("c = %f\n", neuralNetwork.NNCost(nn, ti, to))

	rate := float64(1)

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

	} else {
		for i := 0; i < 13*1000; i++ {
			// to swith between back propagation and finite different method
			if true {
				neuralNetwork.NNBackprop(nn, g, ti, to)
			} else {
				neuralNetwork.NNFiniteDiff(nn, g, 1e-1, ti, to)
			}

			neuralNetwork.NNLearn(nn, g, rate)
			fmt.Printf("%d: c = %f\n", i, neuralNetwork.NNCost(nn, ti, to))
		}
	}

	// i = learning attempt

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
