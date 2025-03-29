"use client";
import { useState } from 'react';

interface RetrainResponse {
  mensaje: string;
  muestras_totales: number;
  metricas: {
    precision: number;
    recall: number;
    f1_score: number;
  };
}

export default function RetrainModel() {
  const [file, setFile] = useState<File | null>(null);
  const [response, setResponse] = useState<RetrainResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Leer el archivo CSV
      const text = await file.text();
      const lines = text.split('\n');
      const headers = lines[0].split(';');
      const data = lines.slice(1).map(line => {
        const values = line.split(';');
        return headers.reduce((obj, header, i) => {
          obj[header.trim()] = values[i]?.trim() || '';
          return obj;
        }, {} as Record<string, string>);
      });

      // Enviar datos al backend
      const res = await fetch('http://localhost:8000/reentrenar/', {
        method: 'POST',
          headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
          },
        body: JSON.stringify({ data }),
      });

      if (!res.ok) throw new Error(await res.text());
      const result = await res.json();
      setResponse(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mt-8 p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Reentrenar Modelo</h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">
            Subir nuevos datos (CSV con columnas: Titulo, Descripcion, Label):
          </label>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-md file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
          />
        </div>

        <button
          type="submit"
          disabled={!file || isLoading}
          className={`px-4 py-2 rounded-md text-white ${(!file || isLoading) ? 'bg-gray-400' : 'bg-green-600 hover:bg-green-700'}`}
        >
          {isLoading ? 'Reentrenando...' : 'Reentrenar Modelo'}
        </button>
      </form>

      {error && (
        <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md">
          Error: {error}
        </div>
      )}

      {response && (
        <div className="mt-6 p-4 bg-gray-50 rounded-md">
          <h3 className="font-bold text-lg">Resultado:</h3>
          <p>{response.mensaje}</p>
          <p>Muestras totales: {response.muestras_totales}</p>
          <div className="mt-2">
            <h4 className="font-semibold">Métricas:</h4>
            <ul className="list-disc pl-5">
              <li>Precisión: {response.metricas.precision}</li>
              <li>Recall: {response.metricas.recall}</li>
              <li>F1 Score: {response.metricas.f1_score}</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}