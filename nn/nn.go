package neuralnetwork

import (
	"fmt"
	"math/rand/v2"

	mlMat "github.com/ml/matrix"
)

type NN struct {
	Count int
	WS    []mlMat.Mat
	BS    []mlMat.Mat
	AS    []mlMat.Mat
}

type ArcDetail struct {
}

type Arc struct {
	Spec []int
	WS   []ArcDetail
	BS   []ArcDetail
	AS   []ArcDetail
}

func NNAlloc(arch *Arc, archCount int, useRandValue bool) NN {
	var nn NN
	nn.Count = archCount - 1

	nn.AS = append(nn.AS, mlMat.Mat{
		Name: "AS0",
		Rows: 1,
		Cols: arch.Spec[0],
		Mat:  GenerateNNColumn(1, arch.Spec[0], false),
	})

	for i := 1; i < archCount; i++ {
		nn.WS = append(nn.WS, mlMat.Mat{
			Rows: nn.AS[i-1].Cols,
			Name: fmt.Sprintf("WS%d", i-1),
			Cols: arch.Spec[i],
			Mat:  GenerateNNColumn(nn.AS[i-1].Cols, arch.Spec[i], useRandValue),
		})
		nn.BS = append(nn.BS, mlMat.Mat{
			Rows: 1,
			Name: fmt.Sprintf("bs%d", i-1),
			Cols: arch.Spec[i],
			Mat:  GenerateNNColumn(1, arch.Spec[i], useRandValue),
		})
		nn.AS = append(nn.AS, mlMat.Mat{
			Rows: 1,
			Name: fmt.Sprintf("AS%d", i),
			Cols: arch.Spec[i],
			Mat:  GenerateNNColumn(1, arch.Spec[i], false),
		})

	}
	return nn
}

func NNPrint(nn NN, name string) {
	fmt.Printf("%s = [\n", name)
	for i := 0; i < nn.Count; i++ {
		nn.WS[i].MatPrint(2)
		nn.BS[i].MatPrint(2)
	}
	fmt.Printf("]\n")
}

func GenerateNNColumn(rows int, cols int, useRandValue bool) mlMat.Matrix {
	value := float64(0)

	var result mlMat.Matrix
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

func NNForward(nn NN) {
	for i := 0; i < nn.Count; i++ {
		mlMat.MatDot(&nn.AS[i+1], nn.AS[i], nn.WS[i])
		mlMat.MatSum(&nn.AS[i+1], nn.BS[i])
		nn.AS[i+1].MatSig()
	}

}

func NNCost(nn NN, ti mlMat.Mat, to mlMat.Mat) float64 {

	nnOutput := nn.AS[len(nn.AS)-1]

	if ti.Rows != to.Rows {
		panic("NNcost:: rows not equal")
	}
	if to.Cols != nnOutput.Cols {
		panic("NNcost:: to cols and nnOutput.cols not equal")
	}

	n := ti.Rows

	c := float64(0)

	for i := 0; i < n; i++ {

		x := mlMat.Mat{
			Rows: 1,
			Name: fmt.Sprintf("ti-%d", i),
			Cols: ti.Cols,
			Mat: mlMat.Matrix{
				ti.Mat[i],
			},
		}

		y := mlMat.Mat{
			Mat: mlMat.Matrix{
				to.Mat[i],
			},
		}

		nn.AS[0].MatCopy(x)
		NNForward(nn)

		q := to.Cols
		for j := 0; j < q; j++ {
			d := nnOutput.Mat[0][j] - y.Mat[0][j]
			c += d * d
		}
	}

	return c / float64(n)
}

func nnZero(nn NN) {
	for i := 0; i < nn.Count; i++ {
		nn.WS[i].MatFil(0)
		nn.BS[i].MatFil(0)
		nn.AS[i].MatFil(0)
	}
	nn.AS[nn.Count].MatFil(0)
}

func NNBackprop(nn NN, g NN, ti mlMat.Mat, to mlMat.Mat) {
	outputLength := len(nn.AS) - 1
	if ti.Rows != to.Rows {
		panic("nnBackdrop:: rows not equal")
	}

	if nn.AS[outputLength].Cols != to.Cols {
		panic("nnBackdrop:: cols not equal")
	}

	nnZero(g)
	n := ti.Rows

	for i := 0; i < n; i++ {
		nn.AS[0].MatCopy(mlMat.Mat{
			Rows: 1,
			Cols: ti.Cols,
			Mat: mlMat.Matrix{
				ti.Mat[i],
			},
		})
		NNForward(nn)

		for j := 0; j < nn.Count; j++ {
			g.AS[j].MatFil(0)
		}

		for j := 0; j < to.Cols; j++ {
			g.AS[len(g.AS)-1].Mat[0][j] = nn.AS[outputLength].Mat[0][j] - to.Mat[i][j]
		}

		for l := nn.Count; l > 0; l-- {
			for j := 0; j < nn.AS[l].Cols; j++ {
				a := nn.AS[l].Mat[0][j]
				da := g.AS[l].Mat[0][j]
				g.BS[l-1].Mat[0][j] += float64(2 * da * a * (1 - a))
				for k := 0; k < nn.AS[l-1].Cols; k++ {
					pa := nn.AS[l-1].Mat[0][k]
					w := nn.WS[l-1].Mat[k][j]
					g.WS[l-1].Mat[k][j] += 2 * da * a * (1 - a) * pa
					g.AS[l-1].Mat[0][k] += 2 * da * a * (1 - a) * w
				}
			}
		}
	}

	for i := 0; i < g.Count; i++ {
		for j := 0; j < g.WS[i].Rows; j++ {
			for k := 0; k < g.WS[i].Cols; k++ {
				g.WS[i].Mat[j][k] /= float64(n)
			}
		}

		for j := 0; j < g.BS[i].Rows; j++ {
			for k := 0; k < g.WS[i].Cols; k++ {
				g.BS[i].Mat[j][k] /= float64(n)
			}
		}
	}
}

func NNFiniteDiff(nn NN, g NN, eps float64, ti mlMat.Mat, to mlMat.Mat) {
	var saved float64
	c := NNCost(nn, ti, to)

	for i := 0; i < nn.Count; i++ {
		for j := 0; j < nn.WS[i].Rows; j++ {
			for k := 0; k < nn.WS[i].Cols; k++ {
				saved = nn.WS[i].Mat[j][k]
				nn.WS[i].Mat[j][k] += eps
				g.WS[i].Mat[j][k] = (NNCost(nn, ti, to) - c) / eps
				nn.WS[i].Mat[j][k] = saved
			}
		}

		for j := 0; j < nn.BS[i].Rows; j++ {
			for k := 0; k < nn.BS[i].Cols; k++ {
				saved = nn.BS[i].Mat[j][k]
				nn.BS[i].Mat[j][k] += eps
				g.BS[i].Mat[j][k] = (NNCost(nn, ti, to) - c) / eps
				nn.BS[i].Mat[j][k] = saved
			}
		}
	}
}

func NNLearn(nn NN, g NN, rate float64) {
	for i := 0; i < nn.Count; i++ {
		for j := 0; j < nn.WS[i].Rows; j++ {
			for k := 0; k < nn.WS[i].Cols; k++ {
				nn.WS[i].Mat[j][k] -= rate * g.WS[i].Mat[j][k]
			}
		}

		for j := 0; j < nn.BS[i].Rows; j++ {
			for k := 0; k < nn.BS[i].Cols; k++ {
				nn.BS[i].Mat[j][k] -= rate * g.BS[i].Mat[j][k]
			}
		}
	}
}
