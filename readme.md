# Gestures Recognition - Eng. Unificada I

> Esse repositório contém os códigos e arquivos para o treinamento, execução do reconhecimento de gestos aplicados a uma interface gráfica. 

---

## Treino

O treino consiste de uma série de comandos para a execução do treinamento utilizando o TensorFlow. Note que é necessário a pasta com as imagens do [Jester](https://20bn.com/datasets/jester). O treinamento pode ser rodado tanto na pasta compactada (muito ineficiente) quanto na pasta descompactada, contudo deve ser modificado no arquivo `train.py`. Demais modificações devem ser feitas no arquivo `config.json`.

---

## Interface

A interface depende da biblioteca Vedo bem como dos arquivos na pasta `mesh`. Os comandos para sua utilização são:

- ↑: Troca de exame
- ←: Volta nos exames passados
- →: Avança nos exames futuros
- s: Seleciona para comparar/compara
- f: Análise de slices
- d: Avança slices (cima)
- a: Volta slices (baixo)
- r: Rotaciona modelo 3d (direira)
- l: Rotaciona modelo 3d (esquerda)