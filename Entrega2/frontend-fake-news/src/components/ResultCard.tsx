import React from 'react';

interface Resultado {
  prediction: string;
  confidence: string;
}

const ResultCard = ({ resultado }: { resultado: Resultado }) => {
  return (
    <div className={`mt-6 p-4 rounded-md border-l-4 ${resultado.prediction === 'falsa' ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'}`}>
      <h2 className="font-bold text-lg">
        Resultado: Noticia <span className={resultado.prediction === 'falsa' ? 'text-red-600' : 'text-green-600'}>
          {resultado.prediction}
        </span>
      </h2>
      <p className="text-gray-600">Confianza: <span className="font-semibold">{resultado.confidence}</span></p>
    </div>
  );
};

export default ResultCard;