package main

import (
	"encoding/json"
	"fmt"
	"os"

	mlMat "github.com/ml/matrix"
	neuralNetwork "github.com/ml/neuralnetwork"
)

func main() {
	t := mlMat.Mat{
		Rows: 4,
		Cols: 3,
		Name: "t",
		Mat:  neuralNetwork.GenerateNNColumn(4, 3, false),
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			row := i*2 + j
			t.Mat[row][0] = float64(i)
			t.Mat[row][1] = float64(j)
			t.Mat[row][2] = float64(i ^ j)
		}
	}

	file, err := os.Create("../xor-mat.json")

	if err != nil {
		fmt.Println("Error creating xor-mat.json:", err)
		panic("error creating xor-mat.json")
	}

	defer file.Close()

	encoder := json.NewEncoder(file)

	if err := encoder.Encode(t); err != nil {
		fmt.Println("Error encoding JSON:", err)
		panic("Error encoding JSON: ")
	}

	fmt.Println("xor-mat JSON generated")
}
