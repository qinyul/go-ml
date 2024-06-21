package main

import (
	"fmt"
)

type Xor struct {
	a0, a1, a2 mat
	w1, b1     mat
	w2, b2     mat
}

var (
	td = Matrix{
		[]float64{0, 0, 0},
		[]float64{0, 1, 1},
		[]float64{1, 0, 1},
		[]float64{1, 1, 0},
	}
	tdx = Matrix{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1},
	}
	tdy = Matrix{
		[]float64{0},
		[]float64{1},
		[]float64{1},
		[]float64{0},
	}
)

func xorAlloc() Xor {
	var m Xor
	x1 := float64(0)
	x2 := float64(1)

	m.a0 = mat{
		rows:   1,
		cols:   2,
		stride: 2,
		mat: Matrix{
			[]float64{
				x1,
				x2,
			},
		},
		name: "x",
	}

	m.w1 = mat{
		rows:   2,
		cols:   2,
		stride: 2,
		mat:    Matrix{},
		name:   "w1",
	}
	m.w1.generateColumn()
	m.b1 = mat{
		rows:   1,
		cols:   2,
		stride: 2,
		mat:    Matrix{},
		name:   "b1",
	}
	m.b1.generateColumn()
	m.a1 = mat{
		rows:   1,
		cols:   2,
		stride: 2,
		mat:    Matrix{},
		name:   "a1",
	}
	m.a1.generateColumn()

	m.w2 = mat{
		rows:   2,
		cols:   1,
		stride: 2,
		mat:    Matrix{},
		name:   "w2",
	}
	m.w2.generateColumn()
	m.b2 = mat{
		rows:   1,
		cols:   1,
		stride: 2,
		mat:    Matrix{},
		name:   "b2",
	}
	m.b2.generateColumn()
	m.a2 = mat{
		rows:   1,
		cols:   1,
		stride: 2,
		mat:    Matrix{},
		name:   "a2",
	}
	m.a2.generateColumn()

	m.w1.matRand(0, 1)
	m.b1.matRand(0, 1)
	m.w2.matRand(0, 1)
	m.b2.matRand(0, 1)

	return m
}

func cost(m Xor, ti mat, to mat) float64 {
	if ti.rows != to.rows {
		panic("cost:: rows not equal")
	}
	if to.cols != m.a2.cols {
		panic("cost:: to cols and m.a2.cols not equal")
	}

	n := len(ti.mat)

	c := float64(0)

	for i := 0; i < n; i++ {

		x := mat{
			rows:   1,
			cols:   m.a0.cols,
			stride: m.a0.stride,
			mat: Matrix{
				ti.mat[i],
			},
		}
		y := mat{
			rows:   1,
			cols:   m.a0.cols,
			stride: m.a0.stride,
			mat: Matrix{
				to.mat[i],
			},
		}

		m.a0.matCopy(x)
		forwardXor(&m)

		q := to.cols
		for j := 0; j < q; j++ {
			d := m.a2.mat[0][j] - y.mat[0][j]
			c += d * d
		}
	}
	// fmt.Printf("c = %f, d = %d\n", c, n)
	return c / float64(n)
}

func forwardXor(m *Xor) {

	matDot(&m.a1, m.a0, m.w1)
	matSum(&m.a1, m.b1)
	m.a1.matSig()

	matDot(&m.a2, m.a1, m.w2)
	matSum(&m.a2, m.b2)
	m.a2.matSig()
}

func finiteDiff(m Xor, g Xor, eps float64, ti mat, to mat) {
	var saved float64

	c := cost(m, ti, to)
	for i := 0; i < m.w1.rows; i++ {
		for j := 0; j < m.w1.cols; j++ {
			saved = m.w1.mat[i][j]
			m.w1.mat[i][j] += eps
			g.w1.mat[i][j] = (cost(m, ti, to) - c) / eps
			m.w1.mat[i][j] = saved
		}
	}

	for i := 0; i < m.b1.rows; i++ {
		for j := 0; j < m.b1.cols; j++ {
			saved = m.b1.mat[i][j]
			m.b1.mat[i][j] += eps
			g.b1.mat[i][j] = (cost(m, ti, to) - c) / eps
			m.b1.mat[i][j] = saved
		}
	}

	for i := 0; i < m.w2.rows; i++ {
		for j := 0; j < m.w2.cols; j++ {
			saved = m.w2.mat[i][j]
			m.w2.mat[i][j] += eps
			g.w2.mat[i][j] = (cost(m, ti, to) - c) / eps
			m.w2.mat[i][j] = saved
		}
	}

	for i := 0; i < m.b2.rows; i++ {
		for j := 0; j < m.b2.cols; j++ {
			saved = m.b2.mat[i][j]
			m.b2.mat[i][j] += eps
			g.b2.mat[i][j] = (cost(m, ti, to) - c) / eps
			m.b2.mat[i][j] = saved
		}
	}
}

func xorLearn(m Xor, g Xor, rate float64) {

	for i := 0; i < m.w1.rows; i++ {
		for j := 0; j < m.w1.cols; j++ {
			m.w1.mat[i][j] -= rate * g.w1.mat[i][j]

		}
	}

	for i := 0; i < m.b1.rows; i++ {
		for j := 0; j < m.b1.cols; j++ {
			m.b1.mat[i][j] -= rate * g.b1.mat[i][j]
		}
	}

	for i := 0; i < m.w2.rows; i++ {
		for j := 0; j < m.w2.cols; j++ {
			m.w2.mat[i][j] -= rate * g.w2.mat[i][j]
		}
	}

	for i := 0; i < m.b2.rows; i++ {
		for j := 0; j < m.b2.cols; j++ {
			m.b2.mat[i][j] -= rate * g.b2.mat[i][j]
		}
	}
}

func main() {

	stride := 3
	n := len(td) * 4 / len(td[0]) * 4 / stride

	ti := mat{
		rows:   n,
		cols:   2,
		name:   "ti",
		stride: stride,
		mat:    tdx,
	}
	to := mat{
		rows:   n,
		cols:   1,
		name:   "ti",
		stride: stride,
		mat:    tdy,
	}

	m := xorAlloc()
	g := xorAlloc()
	eps := 1e-1
	rate := 1e-1

	fmt.Printf("cost = %f\n", cost(m, ti, to))
	for i := 0; i < 12*1000; i++ {
		finiteDiff(m, g, eps, ti, to)
		xorLearn(m, g, rate)
		fmt.Printf("cost = %f\n", cost(m, ti, to))
	}

	fmt.Printf("-------------------\n")
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			m.a0.mat = Matrix{
				[]float64{
					float64(i),
					float64(j),
				},
			}
			forwardXor(&m)
			y := getMatrixValue(m.a2.mat)
			fmt.Printf("%d ^ %d = %f\n", i, j, y)
		}
	}
}
