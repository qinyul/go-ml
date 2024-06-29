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
	neuronRadius    = 25
	layerBorderVpad = 50
	layerBorderHpad = 50
	nnWidth         = SCREEN_WIDTH - 2*layerBorderHpad

	nnHeight = SCREEN_HEIGHT - 2*layerBorderVpad

	frameColor = rl.Red
)

type Visualization struct {
	NN   *neuralNetwork.NN
	Arch neuralNetwork.Arc
	G    *neuralNetwork.NN
	TI   mlMat.Mat
	TO   mlMat.Mat
}

func createFrame() {
	rl.DrawLine(0, 1, SCREEN_WIDTH, 1, frameColor)
	rl.DrawLine(1, 1, 1, SCREEN_HEIGHT, frameColor)
	rl.DrawLine(SCREEN_WIDTH-1, 1, SCREEN_WIDTH-1, SCREEN_HEIGHT, frameColor)
	rl.DrawLine(0, SCREEN_HEIGHT-1, SCREEN_WIDTH, SCREEN_HEIGHT-1, frameColor)
}

func (vs *Visualization) NNRender() {
	nn := vs.NN
	archCount := nn.Count + 1
	nnX := SCREEN_WIDTH/2 - nnWidth/2
	nnY := SCREEN_HEIGHT/2 - nnHeight/2
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

					alpha := math.Floor(255 * mlMat.Sigmoidf(nn.WS[l].Mat[j][0]))
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

func (vs *Visualization) Calculate() {
	rate := 1
	for i := 0; i < 1*1000; i++ {
		neuralNetwork.NNBackprop(*vs.NN, *vs.G, vs.TI, vs.TO)
		neuralNetwork.NNLearn(*vs.NN, *vs.G, float64(rate))

		fmt.Printf("%d: cost = %f\n", i, neuralNetwork.NNCost(*vs.NN, vs.TI, vs.TO))
	}
}

func (vs *Visualization) InitVisualization() {

	rl.InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "machine-learning")

	rl.SetConfigFlags(rl.FlagMsaa4xHint)
	rl.SetTargetFPS(60)

	defer rl.CloseWindow()

	for !rl.WindowShouldClose() {
		rl.ClearBackground(rl.Black)
		vs.Calculate()
		rl.BeginDrawing()
		{
			createFrame()
			vs.NNRender()

		}
		rl.EndDrawing()
	}
}
