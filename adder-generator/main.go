package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"

	mlMat "github.com/ml/matrix"
	neuralNetwork "github.com/ml/neuralnetwork"
)

const (
	BITS = 4
)

func main() {
	n := (1 << BITS)
	rows := n * n

	t := mlMat.Mat{
		Rows: rows,
		Cols: 2*BITS + BITS + 1,
		Name: "t",
		Mat:  neuralNetwork.GenerateNNColumn(rows, 2*BITS+BITS+1, false),
	}
	ti := mlMat.Mat{
		Rows: t.Rows,
		Cols: 2 * BITS,
		Name: "ti",
		Mat:  neuralNetwork.GenerateNNColumn(t.Rows, 2*BITS, false),
	}

	to := mlMat.Mat{
		Rows: t.Rows,
		Cols: BITS + 1,
		Name: "to",
		Mat:  neuralNetwork.GenerateNNColumn(t.Rows, BITS+1, false),
	}
	t.MatPrint(0)
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

	for i := 0; i < t.Rows; i++ {
		for j := 0; j < t.Cols; j++ {
			t.Mat[i][j] = float64(rand.Intn(2))
		}
	}
	file, err := os.Create("../mat.json")

	if err != nil {
		fmt.Println("Error creating mat.json:", err)
		panic("error creating mat.json")
	}

	defer file.Close()

	encoder := json.NewEncoder(file)

	if err := encoder.Encode(t); err != nil {
		fmt.Println("Error encoding JSON:", err)
		panic("Error encoding JSON: ")
	}

	fmt.Println("mat JSON generated")
}
