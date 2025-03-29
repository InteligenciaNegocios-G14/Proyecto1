export async function POST(request: Request) {
  const { Titulo, Descripcion } = await request.json();

  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ Titulo, Descripcion }),
  });

  const data = await response.json();
  return Response.json(data);
}