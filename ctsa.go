package ctsa

/*
#cgo CFLAGS: -I../ctsa/header/ctsa.h
#cgo LDFLAGS: -L../ctsa/Bin -lctsalib -lm
#import "header/ctsa.h"
#import <stdlib.h>
*/
import "C"
import (
	"math"
	"unsafe"
)

// https://github.com/rafat/ctsa

type AutoArimaObject struct {
	ptr C.auto_arima_object
}

type SarimaxObject struct {
	ptr C.sarimax_object
}

type ArimaObject struct {
	ptr C.arima_object
}

type SarimaObject struct {
	ptr C.sarima_object
}

type ArObject struct {
	ptr C.ar_object
}

func AutoArimaInit(pdqmax, PDQmax []int, s, r, N int) *AutoArimaObject {
	cPdqmax := (*C.int)(unsafe.Pointer(&pdqmax[0]))
	cPDQmax := (*C.int)(unsafe.Pointer(&PDQmax[0]))
	obj := C.auto_arima_init(cPdqmax, cPDQmax, C.int(s), C.int(r), C.int(N))
	return &AutoArimaObject{ptr: obj}
}

func SarimaxInit(p, d, q, P, D, Q, s, r, imean, N int) *SarimaxObject {
	obj := C.sarimax_init(C.int(p), C.int(d), C.int(q), C.int(P), C.int(D), C.int(Q), C.int(s), C.int(r), C.int(imean), C.int(N))
	return &SarimaxObject{ptr: obj}
}

func ArimaInit(p, d, q, N int) *ArimaObject {
	obj := C.arima_init(C.int(p), C.int(d), C.int(q), C.int(N))
	return &ArimaObject{ptr: obj}
}

func SarimaInit(p, d, q, s, P, D, Q, N int) *SarimaObject {
	obj := C.sarima_init(C.int(p), C.int(d), C.int(q), C.int(s), C.int(P), C.int(D), C.int(Q), C.int(N))
	return &SarimaObject{ptr: obj}
}

func ArInit(method, N int) *ArObject {
	obj := C.ar_init(C.int(method), C.int(N))
	return &ArObject{ptr: obj}
}

func (obj *AutoArimaObject) Exec(inp, xreg []float64) {
	cInp := (*C.double)(unsafe.Pointer(&inp[0]))
	cXreg := (*C.double)(unsafe.Pointer(&xreg[0]))
	C.auto_arima_exec(obj.ptr, cInp, cXreg)
}

func (obj *SarimaxObject) Exec(inp, xreg []float64) {
	cInp := (*C.double)(unsafe.Pointer(&inp[0]))
	cXreg := (*C.double)(unsafe.Pointer(&xreg[0]))
	C.sarimax_exec(obj.ptr, cInp, cXreg)
}

func (obj *ArimaObject) Exec(x []float64) {
	cX := (*C.double)(unsafe.Pointer(&x[0]))
	C.arima_exec(obj.ptr, cX)
}

func (obj *SarimaObject) Exec(x []float64) {
	cX := (*C.double)(unsafe.Pointer(&x[0]))
	C.sarima_exec(obj.ptr, cX)
}

func (obj *ArObject) Exec(inp []float64) {
	cInp := (*C.double)(unsafe.Pointer(&inp[0]))
	C.ar_exec(obj.ptr, cInp)
}

func (obj *ArimaObject) Predict(inp []float64, L int, xpred, amse []float64) {
	cInp := (*C.double)(unsafe.Pointer(&inp[0]))
	cXpred := (*C.double)(unsafe.Pointer(&xpred[0]))
	cAmse := (*C.double)(unsafe.Pointer(&amse[0]))
	C.arima_predict(obj.ptr, cInp, C.int(L), cXpred, cAmse)
}

func (obj *SarimaObject) Predict(inp []float64, L int, xpred, amse []float64) {
	cInp := (*C.double)(unsafe.Pointer(&inp[0]))
	cXpred := (*C.double)(unsafe.Pointer(&xpred[0]))
	cAmse := (*C.double)(unsafe.Pointer(&amse[0]))
	C.sarima_predict(obj.ptr, cInp, C.int(L), cXpred, cAmse)
}

func (obj *SarimaxObject) Predict(inp, xreg []float64, L int, newxreg, xpred, amse []float64) {
	cInp := (*C.double)(unsafe.Pointer(&inp[0]))
	cXreg := (*C.double)(unsafe.Pointer(&xreg[0]))
	cNewxreg := (*C.double)(unsafe.Pointer(&newxreg[0]))
	cXpred := (*C.double)(unsafe.Pointer(&xpred[0]))
	cAmse := (*C.double)(unsafe.Pointer(&amse[0]))
	C.sarimax_predict(obj.ptr, cInp, cXreg, C.int(L), cNewxreg, cXpred, cAmse)
}

func (obj *AutoArimaObject) Predict(inp, xreg []float64, L int, newxreg, xpred, amse []float64) {
	cInp := (*C.double)(unsafe.Pointer(&inp[0]))
	cXreg := (*C.double)(unsafe.Pointer(&xreg[0]))
	cNewxreg := (*C.double)(unsafe.Pointer(&newxreg[0]))
	cXpred := (*C.double)(unsafe.Pointer(&xpred[0]))
	cAmse := (*C.double)(unsafe.Pointer(&amse[0]))
	C.auto_arima_predict(obj.ptr, cInp, cXreg, C.int(L), cNewxreg, cXpred, cAmse)
}

func (obj *ArObject) Predict(inp []float64, L int, xpred, amse []float64) {
	cInp := (*C.double)(unsafe.Pointer(&inp[0]))
	cXpred := (*C.double)(unsafe.Pointer(&xpred[0]))
	cAmse := (*C.double)(unsafe.Pointer(&amse[0]))
	C.ar_predict(obj.ptr, cInp, C.int(L), cXpred, cAmse)
}

func (obj *ArimaObject) SetMethod(value int) {
	C.arima_setMethod(obj.ptr, C.int(value))
}

func (obj *SarimaObject) SetMethod(value int) {
	C.sarima_setMethod(obj.ptr, C.int(value))
}

func (obj *AutoArimaObject) SetMethod(value int) {
	C.auto_arima_setMethod(obj.ptr, C.int(value))
}

func (obj *SarimaxObject) SetMethod(value int) {
	C.sarimax_setMethod(obj.ptr, C.int(value))
}

func (obj *ArimaObject) SetOptMethod(value int) {
	C.arima_setOptMethod(obj.ptr, C.int(value))
}

func (obj *SarimaObject) SetOptMethod(value int) {
	C.sarima_setOptMethod(obj.ptr, C.int(value))
}

func (obj *SarimaxObject) SetOptMethod(value int) {
	C.sarimax_setOptMethod(obj.ptr, C.int(value))
}

func (obj *AutoArimaObject) SetOptMethod(value int) {
	C.auto_arima_setOptMethod(obj.ptr, C.int(value))
}

func (obj *ArimaObject) Vcov(vcov []float64) {
	cVcov := (*C.double)(unsafe.Pointer(&vcov[0]))
	C.arima_vcov(obj.ptr, cVcov)
}

func (obj *SarimaObject) Vcov(vcov []float64) {
	cVcov := (*C.double)(unsafe.Pointer(&vcov[0]))
	C.sarima_vcov(obj.ptr, cVcov)
}

func (obj *SarimaxObject) Vcov(vcov []float64) {
	cVcov := (*C.double)(unsafe.Pointer(&vcov[0]))
	C.sarimax_vcov(obj.ptr, cVcov)
}

func (obj *AutoArimaObject) SetApproximation(approximation int) {
	C.auto_arima_setApproximation(obj.ptr, C.int(approximation))
}

func (obj *AutoArimaObject) SetStepwise(stepwise int) {
	C.auto_arima_setStepwise(obj.ptr, C.int(stepwise))
}

func (obj *AutoArimaObject) SetStationary(stationary int) {
	C.auto_arima_setStationary(obj.ptr, C.int(stationary))
}

func (obj *AutoArimaObject) SetSeasonal(seasonal int) {
	C.auto_arima_setSeasonal(obj.ptr, C.int(seasonal))
}

func (obj *AutoArimaObject) SetStationarityParameters(test string, alpha float64, testType string) {
	cTest := C.CString(test)
	defer C.free(unsafe.Pointer(cTest))
	cTestType := C.CString(testType)
	defer C.free(unsafe.Pointer(cTestType))
	C.auto_arima_setStationarityParameters(obj.ptr, cTest, C.double(alpha), cTestType)
}

func (obj *AutoArimaObject) SetSeasonalParameters(test string, alpha float64) {
	cTest := C.CString(test)
	defer C.free(unsafe.Pointer(cTest))
	C.auto_arima_setSeasonalParameters(obj.ptr, cTest, C.double(alpha))
}

func (obj *AutoArimaObject) SetVerbose(verbose int) {
	C.auto_arima_setVerbose(obj.ptr, C.int(verbose))
}

func (obj *ArimaObject) Summary() {
	C.arima_summary(obj.ptr)
}

func (obj *SarimaObject) Summary() {
	C.sarima_summary(obj.ptr)
}

func (obj *SarimaxObject) Summary() {
	C.sarimax_summary(obj.ptr)
}

func (obj *AutoArimaObject) Summary() {
	C.auto_arima_summary(obj.ptr)
}

func (obj *ArObject) Summary() {
	C.ar_summary(obj.ptr)
}

func ArEstimate(x []float64, N, method int) int {
	cX := (*C.double)(unsafe.Pointer(&x[0]))
	return int(C.ar_estimate(cX, C.int(N), C.int(method)))
}

func ModelEstimate(x []float64, N, d, pmax, h int) {
	cX := (*C.double)(unsafe.Pointer(&x[0]))
	C.model_estimate(cX, C.int(N), C.int(d), C.int(pmax), C.int(h))
}

func Pacf(vec []float64, N int, par []float64, M int) {
	cVec := (*C.double)(unsafe.Pointer(&vec[0]))
	cPar := (*C.double)(unsafe.Pointer(&par[0]))
	C.pacf(cVec, C.int(N), cPar, C.int(M))
}

func PacfOpt(vec []float64, N, method int, par []float64, M int) {
	cVec := (*C.double)(unsafe.Pointer(&vec[0]))
	cPar := (*C.double)(unsafe.Pointer(&par[0]))
	C.pacf_opt(cVec, C.int(N), C.int(method), cPar, C.int(M))
}

func Acvf(vec []float64, N int, par []float64, M int) {
	cVec := (*C.double)(unsafe.Pointer(&vec[0]))
	cPar := (*C.double)(unsafe.Pointer(&par[0]))
	C.acvf(cVec, C.int(N), cPar, C.int(M))
}

func AcvfOpt(vec []float64, N, method int, par []float64, M int) {
	cVec := (*C.double)(unsafe.Pointer(&vec[0]))
	cPar := (*C.double)(unsafe.Pointer(&par[0]))
	C.acvf_opt(cVec, C.int(N), C.int(method), cPar, C.int(M))
}

func Acvf2Acf(acf []float64, M int) {
	cAcf := (*C.double)(unsafe.Pointer(&acf[0]))
	C.acvf2acf(cAcf, C.int(M))
}

func (obj *ArimaObject) Free() {
	C.arima_free(obj.ptr)
}

func (obj *SarimaObject) Free() {
	C.sarima_free(obj.ptr)
}

func (obj *SarimaxObject) Free() {
	C.sarimax_free(obj.ptr)
}

func (obj *AutoArimaObject) Free() {
	C.auto_arima_free(obj.ptr)
}

func (obj *ArObject) Free() {
	C.ar_free(obj.ptr)
}

func Yw(x []float64, N, p int, phi, vari []float64) {
	cX := (*C.double)(unsafe.Pointer(&x[0]))
	cPhi := (*C.double)(unsafe.Pointer(&phi[0]))
	cVar := (*C.double)(unsafe.Pointer(&vari[0]))
	C.yw(cX, C.int(N), C.int(p), cPhi, cVar)
}

func Burg(x []float64, N, p int, phi, vari []float64) {
	cX := (*C.double)(unsafe.Pointer(&x[0]))
	cPhi := (*C.double)(unsafe.Pointer(&phi[0]))
	cVar := (*C.double)(unsafe.Pointer(&vari[0]))
	C.burg(cX, C.int(N), C.int(p), cPhi, cVar)
}

func Hr(x []float64, N, p, q int, phi, theta, vari []float64) {
	cX := (*C.double)(unsafe.Pointer(&x[0]))
	cPhi := (*C.double)(unsafe.Pointer(&phi[0]))
	cTheta := (*C.double)(unsafe.Pointer(&theta[0]))
	cVar := (*C.double)(unsafe.Pointer(&vari[0]))
	C.hr(cX, C.int(N), C.int(p), C.int(q), cPhi, cTheta, cVar)
}

// Function to calculate AIC
func calculateAIC(n int, rss float64, k int) float64 {
	return float64(n)*math.Log(rss/float64(n)) + 2*float64(k)
}

// Function to calculate RSS (Residual Sum of Squares)
func calculateRSS(residuals *C.double, n int) float64 {
	var rss float64
	for i := 0; i < n; i++ {
		res := *(*C.double)(unsafe.Pointer(uintptr(unsafe.Pointer(residuals)) + uintptr(i)*unsafe.Sizeof(*residuals)))
		rss += float64(res * res)
	}
	return rss
}

// Function to difference the series
func difference(series []float64, d int) []float64 {
	if d == 0 {
		return series
	}

	diff := make([]float64, len(series)-d)
	for i := d; i < len(series); i++ {
		diff[i-d] = series[i] - series[i-d]
	}
	return diff
}

// Function to find the best ARIMA model
func CCAutoArima(data []float64, maxP, maxD, maxQ int) (int, int, int, float64) {
	n := len(data)
	bestAIC := math.Inf(1)
	var bestP, bestD, bestQ int

	for p := 0; p <= maxP; p++ {
		for d := 0; d <= maxD; d++ {
			for q := 0; q <= maxQ; q++ {
				diffData := difference(data, d)
				model := C.arima_init(C.int(p), C.int(d), C.int(q), C.int(len(diffData)))
				defer C.arima_free(model)

				cData := (*C.double)(unsafe.Pointer(&diffData[0]))
				C.arima_exec(model, cData)

				// Calculate RSS from residuals
				rss := calculateRSS(model.res, int(model.Nused))

				// Calculate AIC
				aic := calculateAIC(n, rss, p+q+1)

				if aic < bestAIC {
					bestAIC = aic
					bestP = p
					bestD = d
					bestQ = q
				}
			}
		}
	}

	return bestP, bestD, bestQ, bestAIC
}
