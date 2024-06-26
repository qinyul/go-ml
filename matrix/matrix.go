package matrix

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

type Matrix [][]float64

type Mat struct {
	Rows   int    `json:"rows"`
	Cols   int    `json:"cols"`
	Name   string `json:"name"`
	Stride int    `json:"stride"`
	Mat    Matrix `json:"mat"`
}

func Sigmoidf(x float64) float64 {
	return float64(1) / (float64(1) + math.Exp(-x))
}

func MatDot(dst *Mat, a Mat, b Mat) {
	if a.Cols != b.Rows {
		panic("a.cols and b.rows not equal")
	}

	if dst.Rows != a.Rows {
		panic("dst.rows and a.rows not equal")
	}

	if dst.Cols != b.Cols {
		panic("dst.cols and b.cols not equal")
	}

	n := a.Cols
	for i := 0; i < dst.Rows; i++ {
		for j := 0; j < dst.Cols; j++ {
			dst.Mat[i][j] = 0
			for k := 0; k < n; k++ {
				dst.Mat[i][j] += a.Mat[i][k] * b.Mat[k][j]
			}
		}
	}
}

func (m *Mat) MatFil(x float64) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Mat[i][j] = x
		}
	}
}

func (m *Mat) MatCopy(src Mat) {
	if m.Rows != src.Rows {
		fmt.Printf("m.rows = %d src.rows = %d\n", m.Rows, src.Rows)
		panic("matCopy:: rows not equal")
	}

	if m.Cols != src.Cols {
		src.MatPrint(0)
		fmt.Printf("m.cols = %d, src.cols = %d\n", m.Cols, src.Cols)
		panic("matCopy:: cols not equal")
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Mat[i][j] = src.Mat[i][j]
		}
	}
}

func MatSum(dst *Mat, a Mat) {
	if dst.Rows == a.Rows && dst.Cols == a.Cols {
		for i := 0; i < dst.Rows; i++ {
			for j := 0; j < dst.Cols; j++ {
				dst.Mat[i][j] += a.Mat[i][j]
			}
		}
	} else {
		panic("invalid matrix, dst and a is not equal")
	}
}

func (m *Mat) MatSig() {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Mat[i][j] = Sigmoidf(m.Mat[i][j])
		}
	}
}

func (m *Mat) MatPrint(padding int) {
	fmt.Printf("%*s%s = [\n", padding, "", m.Name)
	for i := 0; i < len(m.Mat); i++ {
		for j := 0; j < len(m.Mat[i]); j++ {
			fmt.Printf("%*s    ", padding, "")
			fmt.Printf("%f ", m.Mat[i][j])
		}
		fmt.Printf("\n")
	}
	fmt.Printf("%*s]\n", padding, "")
}

func LoadMatJSON(filePath string) Mat {
	file, err := os.Open(filePath)
	if err != nil {
		fmt.Println("Error opening mat json file:", err)
		panic("Error opening mat json file")
	}

	defer file.Close()

	scanner := bufio.NewScanner(file)

	var jsonData []byte
	for scanner.Scan() {
		jsonData = append(jsonData, scanner.Bytes()...)
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error scanning file:", err)
		panic("error scanning file")
	}

	var mat Mat

	if err := json.Unmarshal(jsonData, &mat); err != nil {
		fmt.Println("Error unmarshalling JSON:", err)
		panic("error unmarshalling json")
	}

	return mat
}
