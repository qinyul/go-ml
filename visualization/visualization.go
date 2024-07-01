package visualization

import (
	"fmt"
	"math"

	rl "github.com/gen2brain/raylib-go/raylib"
	mlMat "github.com/ml/matrix"
	neuralNetwork "github.com/ml/neuralnetwork"
)

const (
	SCREEN_HEIGHT = 700
	SCREEN_WIDTH  = 700
)

var (
	layerBorderVpad = 50
	layerBorderHpad = 50

	frameColor = rl.Red
)

type Visualization struct {
	NN   *neuralNetwork.NN
	Arch neuralNetwork.Arc
	G    *neuralNetwork.NN
	TI   mlMat.Mat
	TO   mlMat.Mat
	RX   int
	RY   int
	RW   int
	RH   int
	PLOT []float64
}

func createFrame() {
	rl.DrawLine(0, 1, SCREEN_WIDTH, 1, frameColor)
	rl.DrawLine(1, 1, 1, SCREEN_HEIGHT, frameColor)
	rl.DrawLine(SCREEN_WIDTH-1, 1, SCREEN_WIDTH-1, SCREEN_HEIGHT, frameColor)
	rl.DrawLine(0, SCREEN_HEIGHT-1, SCREEN_WIDTH, SCREEN_HEIGHT-1, frameColor)
}

func (vs *Visualization) NNRender() {
	nn := vs.NN
	neuronRadius := float64(float64(vs.RH) * 0.04)
	archCount := nn.Count + 1
	nnWidth := vs.RW - 2*layerBorderHpad
	nnHeight := vs.RH - 2*layerBorderVpad
	nnX := vs.RX + vs.RW/2 - nnWidth/2
	nnY := vs.RY + vs.RH/2 - nnHeight/2
	layerHpad := nnWidth / archCount

	for l := 0; l < archCount; l++ {
		layerVpad1 := nnHeight / nn.AS[l].Cols

		for i := 0; i < nn.AS[l].Cols; i++ {
			cx1 := int32(nnX + l*layerHpad + layerHpad/2)
			cy1 := int32(nnY + i*layerVpad1 + layerVpad1/2)

			if l+1 < archCount {
				layerVpad2 := nnHeight / nn.AS[l+1].Cols
				for j := 0; j < nn.AS[l+1].Cols; j++ {
					cx2 := nnX + (l+1)*layerHpad + layerHpad/2
					cy2 := nnY + j*layerVpad2 + layerVpad2/2

					rowIndex := j
					if j >= len(nn.WS[l].Mat) {
						rowIndex = 0
					}

					alpha := math.Floor(255 * mlMat.Sigmoidf(nn.WS[l].Mat[rowIndex][0]))
					connectionColor := rl.SkyBlue
					connectionColor.R = 255
					connectionColor.G = uint8(int(alpha) * l * 20)
					connectionColor.B = uint8(int(alpha) << (8 * 3))
					rl.DrawLine(cx1, cy1, int32(cx2), int32(cy2), connectionColor)
				}
			}
			if l > 0 {
				s := math.Floor(255 * mlMat.Sigmoidf(nn.BS[l-1].Mat[0][i]))
				neuronColor := rl.Maroon
				neuronColor.R = 255
				neuronColor.G = uint8(int(s) * l * 20)
				neuronColor.B = uint8(int(s) << (8 * 3))
				rl.DrawCircle(cx1, cy1, float32(neuronRadius), neuronColor)
			} else {
				rl.DrawCircle(cx1, cy1, float32(neuronRadius), rl.DarkGray)
			}
		}
	}

}

func (vs *Visualization) Calculate(i *int) {
	rate := 1
	maxEpoch := 5000
	if *i < maxEpoch {
		neuralNetwork.NNBackprop(*vs.NN, *vs.G, vs.TI, vs.TO)
		neuralNetwork.NNLearn(*vs.NN, *vs.G, float64(rate))
		*i += 1
		vs.PLOT = append(vs.PLOT, neuralNetwork.NNCost(*vs.NN, vs.TI, vs.TO))
		fmt.Printf("cost = %f \n", neuralNetwork.NNCost(*vs.NN, vs.TI, vs.TO))
	} else {
		fmt.Println("Learning process done")
	}

	ep := fmt.Sprintf("Epoch: %d/%d Rate:%d", *i, maxEpoch, rate)
	rl.DrawText(ep, 0, 0, SCREEN_HEIGHT*0.04, rl.RayWhite)
}

func (vs *Visualization) costPlotMinMax() (float64, float64) {
	max := math.SmallestNonzeroFloat64
	min := math.MaxFloat64

	for i := 0; i < len(vs.PLOT)-1; i++ {
		if max < vs.PLOT[i] {
			max = vs.PLOT[i]
		}
		if min > vs.PLOT[i] {
			min = vs.PLOT[i]
		}
	}

	return min, max
}

func (vs *Visualization) plotCost() {
	min, max := vs.costPlotMinMax()
	plotCount := len(vs.PLOT) - 1
	n := plotCount
	if min > 0 {
		min = 0
	}

	if n < 1000 {
		n = 1000
	}
	for i := 0; i+1 < plotCount; i++ {
		x1 := float64(vs.RX) + float64(vs.RW)/float64(n)*float64(i)
		y1 := float64(vs.RY) + (1-(float64(vs.PLOT[i])-min)/(max-min))*float64(vs.RH)
		x2 := float64(vs.RX) + float64(vs.RW)/float64(n)*float64(i) + 1
		y2 := float64(vs.RY) + (1-(float64(vs.PLOT[i])-min)/(max-min))*float64(vs.RH)
		rl.DrawLineEx(rl.Vector2{X: float32(x1), Y: float32(y1)}, rl.Vector2{X: float32(x2), Y: float32(y2)}, float32(float32(vs.RH)*0.004), rl.Red)
	}
}

func (vs *Visualization) InitVisualization() {
	initialRx := vs.RX
	rl.InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "machine-learning")

	rl.SetConfigFlags(rl.FlagMsaa4xHint)
	rl.SetTargetFPS(60)

	defer rl.CloseWindow()

	epoch := 0

	for !rl.WindowShouldClose() {
		rl.ClearBackground(rl.Black)
		rl.BeginDrawing()
		{
			vs.Calculate(&epoch)

			createFrame()

			vs.RX = 0
			vs.plotCost()

			vs.RX = initialRx
			vs.NNRender()

		}
		rl.EndDrawing()
	}
}
