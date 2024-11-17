'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Camera, Upload } from 'lucide-react'

export default function ReconocimientoSenas() {
  const [modelo, setModelo] = useState(null)
  const [camaraActiva, setCamaraActiva] = useState(false)
  const [gestoReconocido, setGestoReconocido] = useState('')
  const [historial, setHistorial] = useState([])
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  const cargarModelo = (evento) => {
    // Aquí iría la lógica para cargar el modelo con TensorFlow.js
    console.log('Modelo cargado:', evento.target.files[0].name)
    setModelo(evento.target.files[0].name)
  }

  const activarCamara = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      videoRef.current.srcObject = stream
      setCamaraActiva(true)
    } catch (error) {
      console.error('Error al acceder a la cámara:', error)
    }
  }

  useEffect(() => {
    if (camaraActiva && modelo) {
      const intervalo = setInterval(() => {
        // Aquí iría la lógica para procesar el video y reconocer gestos
        const gestoAleatorio = ['Hola', 'Gracias', 'Por favor', 'Adiós'][Math.floor(Math.random() * 4)]
        setGestoReconocido(gestoAleatorio)
        setHistorial(prev => [gestoAleatorio, ...prev].slice(0, 5))
      }, 2000)

      return () => clearInterval(intervalo)
    }
  }, [camaraActiva, modelo])

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-3xl font-bold mb-6 text-center text-gray-800">Reconocimiento de Lenguaje de Señas</h1>
      
      <Card className="mb-6">
        <CardContent className="p-4">
          <Label htmlFor="cargar-modelo" className="mb-2 block">Cargar Modelo</Label>
          <div className="flex items-center space-x-2">
            <Input id="cargar-modelo" type="file" accept=".json,.bin" onChange={cargarModelo} className="flex-grow" />
            <Button variant="outline"><Upload className="mr-2 h-4 w-4" /> Cargar</Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardContent className="p-4">
            <h2 className="text-xl font-semibold mb-4">Captura de Video</h2>
            {camaraActiva ? (
              <div className="relative aspect-video">
                <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover rounded-lg" />
                <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full" />
              </div>
            ) : (
              <Button onClick={activarCamara} className="w-full py-8" disabled={!modelo}>
                <Camera className="mr-2 h-5 w-5" /> Activar Cámara
              </Button>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <h2 className="text-xl font-semibold mb-4">Resultados en Vivo</h2>
            <div className="bg-gray-100 p-4 rounded-lg min-h-[100px] flex items-center justify-center">
              <p className="text-2xl font-bold text-gray-700">{gestoReconocido || 'Esperando gesto...'}</p>
            </div>
            
            <h3 className="text-lg font-semibold mt-6 mb-2">Historial de Reconocimientos</h3>
            <ul className="space-y-2">
              {historial.map((gesto, index) => (
                <li key={index} className="bg-gray-50 p-2 rounded">{gesto}</li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}