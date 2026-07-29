from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/saludar', methods=['POST'])
def saludar():
    # Obtenemos el nombre del formulario
    nombre_usuario = request.form.get('input_nombre')
    # Redirigimos a la página de saludo pasando el nombre
    return render_template('saludo.html', nombre=nombre_usuario)

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/otra-pagina')
def otra_pagina():
    return '''
<!DOCTYPE html>
<html lang="es">
<head>
    <title>Otra Página</title>
</head>
<body>
    <h1>Bienvenido a otra página</h1>
    <p>Esta es otra página de la aplicación Flask.</p>
    <a href="/">Volver a la página de inicio</a>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)