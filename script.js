document.addEventListener('DOMContentLoaded', () => {
    // --- 1. Константы и состояние ---
    const boardElement = document.getElementById('game-board');
    const cells = document.querySelectorAll('.cell');
    const statusElement = document.getElementById('status');
    const resetButton = document.getElementById('reset-button');
    const nnHeader = document.getElementById('nn-header');
    const nnLegend = document.getElementById('nn-legend');
    const nnVisualizationContainer = document.getElementById('nn-visualization');

    const PLAYER_X = 'X';
    const PLAYER_O = 'O'; // ИИ
    let currentPlayer = PLAYER_X;
    let board = [0, 0, 0, 0, 0, 0, 0, 0, 0]; // 0: пусто, 1: X, -1: O
    let gameActive = true;

    // --- 2. Настройка ИИ (Brain.js) ---
    const net = new brain.NeuralNetwork({
        inputSize: 9, // 9 входных нейронов (клетки)
        hiddenLayers: [15], // Увеличенный скрытый слой
        outputSize: 9, // 9 выходных нейронов (вероятности ходов)
        activation: 'sigmoid'
    });
    let trainingData = [];

    // --- 3. Генерация обучающих данных для ИИ ---
    const winPatterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], // строки
        [0, 3, 6], [1, 4, 7], [2, 5, 8], // столбцы
        [0, 4, 8], [2, 4, 6]  // диагонали
    ];

    function getBestMove(board, player) {
        const opponent = player === 1 ? -1 : 1;
        for (let i = 0; i < board.length; i++) {
            if (board[i] === 0) {
                const tempBoard = [...board];
                tempBoard[i] = player;
                if (checkWinner(tempBoard) === player) {
                    return i;
                }
            }
        }
        for (let i = 0; i < board.length; i++) {
            if (board[i] === 0) {
                const tempBoard = [...board];
                tempBoard[i] = opponent;
                if (checkWinner(tempBoard) === opponent) {
                    return i;
                }
            }
        }
        if (board[4] === 0) return 4;
        const corners = [0, 2, 6, 8];
        const availableCorners = corners.filter(i => board[i] === 0);
        if (availableCorners.length > 0) {
            return availableCorners[Math.floor(Math.random() * availableCorners.length)];
        }
        const sides = [1, 3, 5, 7];
        const availableSides = sides.filter(i => board[i] === 0);
        if (availableSides.length > 0) {
            return availableSides[Math.floor(Math.random() * availableSides.length)];
        }
        return -1;
    }

    function generateTrainingData() {
        const data = new Set();
        for (let i = 0; i < 15000; i++) {
            let simBoard = [0, 0, 0, 0, 0, 0, 0, 0, 0];
            let simPlayer = 1;
            while (true) {
                const winner = checkWinner(simBoard);
                const availableMoves = simBoard.map((v, idx) => v === 0 ? idx : -1).filter(idx => idx !== -1);
                if (winner || availableMoves.length === 0) {
                    break;
                }
                let move;
                if (simPlayer === 1) {
                    move = availableMoves[Math.floor(Math.random() * availableMoves.length)];
                } else {
                    const boardStateBeforeMove = [...simBoard];
                    const bestMove = getBestMove(boardStateBeforeMove, -1);
                    move = bestMove;
                    if (move !== -1) {
                        const output = Array(9).fill(0);
                        output[bestMove] = 1;
                        addBoardVariations(boardStateBeforeMove, output, data);
                    }
                }
                if (move !== -1 && simBoard[move] === 0) {
                    simBoard[move] = simPlayer;
                }
                simPlayer *= -1;
            }
        }
        trainingData = Array.from(data).map(item => JSON.parse(item));
    }

    function addBoardVariations(board, output, dataSet) {
        let currentBoard = [...board];
        let currentOutput = [...output];
        for (let i = 0; i < 4; i++) {
            dataSet.add(JSON.stringify({ input: currentBoard, output: currentOutput }));
            const reflectedBoard = reflectBoard(currentBoard);
            const reflectedOutput = reflectBoard(currentOutput);
            dataSet.add(JSON.stringify({ input: reflectedBoard, output: reflectedOutput }));
            currentBoard = rotateBoard(currentBoard);
            currentOutput = rotateBoard(currentOutput);
        }
    }

    function rotateBoard(board) {
        const newBoard = Array(9).fill(0);
        const mapping = [6, 3, 0, 7, 4, 1, 8, 5, 2];
        for (let i = 0; i < 9; i++) {
            newBoard[i] = board[mapping[i]];
        }
        return newBoard;
    }

    function reflectBoard(board) {
        const newBoard = Array(9).fill(0);
        const mapping = [2, 1, 0, 5, 4, 3, 8, 7, 6];
        for (let i = 0; i < 9; i++) {
            newBoard[i] = board[mapping[i]];
        }
        return newBoard;
    }

    async function trainAI() {
        gameActive = false;
        statusElement.textContent = 'ИИ обучается... (0%)';
        await new Promise(resolve => setTimeout(resolve, 10));
        generateTrainingData();
        console.log(`Обучение на ${trainingData.length} примерах.`);
        console.log('Структура сети перед обучением:', net.sizes);

        const trainingOptions = {
            iterations: 500,
            log: (stats) => {
                const progress = Math.round((stats.iterations / 500) * 100);
                statusElement.textContent = `ИИ обучается... (${progress}%)`;
                console.log(stats);
            },
            logPeriod: 50,
            errorThresh: 0.01
        };

        try {
            await net.trainAsync(trainingData, trainingOptions);
            console.log('Структура сети после обучения:', net.sizes);
            statusElement.textContent = 'Ваш ход (X)';
            console.log('Обучение ИИ завершено.');
            gameActive = true;

            // --- Создание и начальная настройка визуализации ---
            console.log('Создание визуализации сети...');
            const svg = brain.utilities.toSVG(net);
            if (!svg) {
                console.error('Ошибка: SVG не сгенерирован.');
                statusElement.textContent = 'Ошибка визуализации нейронной сети.';
                return;
            }
            console.log('SVG сгенерирован:', svg.substring(0, 100) + '...');
            nnVisualizationContainer.innerHTML = svg;
            nnHeader.style.display = 'block';
            nnLegend.style.display = 'block';
            updateVisualization(board, Array(9).fill(0));
        } catch (error) {
            console.error('Ошибка при создании визуализации:', error);
            statusElement.textContent = 'Ошибка визуализации нейронной сети.';
        }
    }

    function updateVisualization(currentBoard, networkOutput) {
        const svg = nnVisualizationContainer.querySelector('svg');
        if (!svg) {
            console.error('SVG не найден в контейнере.');
            return;
        }

        console.log('Обновление визуализации:', { currentBoard, networkOutput }); // Отладка

        const colorEmpty = '#ffffff';
        const colorX = '#ff4136';
        const colorO = '#0074d9';
        const colorOccupied = '#cccccc';
        const colorProbLow = '#e6f7ff';

        const lerpColor = (a, b, amount) => {
            const ar = a >> 16, ag = a >> 8 & 0xff, ab = a & 0xff,
                  br = b >> 16, bg = b >> 8 & 0xff, bb = b & 0xff,
                  rr = Math.round(ar + amount * (br - ar)),
                  rg = Math.round(ag + amount * (bg - ag)),
                  rb = Math.round(ab + amount * (bb - ab));
            return `#${(1 << 24 | rr << 16 | rg << 8 | rb).toString(16).slice(1)}`;
        };

        const hexToDec = (hex) => parseInt(hex.replace("#", ""), 16);
        const decProbLow = hexToDec(colorProbLow);
        const decProbHigh = hexToDec(colorO);

        // Проверка и обновление входного слоя
        const inputNodes = svg.querySelectorAll('.layer.input .node');
        if (inputNodes.length !== 9) {
            console.warn(`Ожидалось 9 входных нейронов, найдено ${inputNodes.length}`);
        }
        inputNodes.forEach((node, index) => {
            const circle = node.querySelector('circle');
            if (!circle) {
                console.warn(`Круг не найден для входного нейрона ${index}`);
                return;
            }
            const boardValue = currentBoard[index];
            circle.style.fill = boardValue === 1 ? colorX : boardValue === -1 ? colorO : colorEmpty;
            console.log(`Входной нейрон ${index}: fill = ${circle.style.fill}`); // Отладка
        });

        // Проверка и обновление выходного слоя
        const outputNodes = svg.querySelectorAll('.layer.output .node');
        if (outputNodes.length !== 9) {
            console.warn(`Ожидалось 9 выходных нейронов, найдено ${outputNodes.length}`);
        }
        outputNodes.forEach((node, index) => {
            const circle = node.querySelector('circle');
            if (!circle) {
                console.warn(`Круг не найден для выходного нейрона ${index}`);
                return;
            }
            if (currentBoard[index] !== 0) {
                circle.style.fill = colorOccupied;
            } else {
                const probability = networkOutput[index] || 0;
                const newColor = lerpColor(decProbLow, decProbHigh, probability);
                circle.style.fill = newColor;
                console.log(`Выходной нейрон ${index}: probability = ${probability}, fill = ${newColor}`); // Отладка
            }
        });

        // Принудительное обновление SVG
        svg.setAttribute('width', svg.getAttribute('width') || '100%');
        svg.setAttribute('height', svg.getAttribute('height') || 'auto');
    }

    function handleCellClick(e) {
        if (!gameActive || currentPlayer !== PLAYER_X) return;
        const index = parseInt(e.target.dataset.index);
        if (board[index] === 0) {
            makeMove(index, PLAYER_X);
            if (gameActive) {
                currentPlayer = PLAYER_O;
                updateStatus();
                setTimeout(aiMove, 500);
            }
        }
    }

    function makeMove(index, player) {
        if (board[index] !== 0 || !gameActive) return;
        board[index] = (player === PLAYER_X) ? 1 : -1;
        cells[index].textContent = player;
        cells[index].classList.add(player.toLowerCase());
        const winner = checkWinner(board);
        if (winner) {
            gameActive = false;
            statusElement.textContent = (winner === 1) ? 'Вы победили! (X)' : 'ИИ победил! (O)';
        } else if (!board.includes(0)) {
            gameActive = false;
            statusElement.textContent = 'Ничья!';
        }
    }

    function aiMove() {
        if (!gameActive) return;
        const output = net.run(board);
        console.log('Выходные данные сети:', output); // Отладка
        updateVisualization(board, output);
        let maxProb = -1;
        let move = -1;
        for (let i = 0; i < output.length; i++) {
            if (board[i] === 0 && output[i] > maxProb) {
                maxProb = output[i];
                move = i;
            }
        }
        // Выводим вероятный ход ИИ только для свободной клетки
        const aiMoveInfo = document.getElementById('ai-move-info');
        if (move !== -1) {
            aiMoveInfo.innerHTML = `<span style='font-size:1.5em;margin-right:8px;'>🤖</span>Вероятный ход ИИ: свободная клетка ${move + 1}, вероятность ${maxProb.toFixed(3)}`;
            // Делаем ход ИИ с задержкой для наглядности
            setTimeout(() => {
                makeMove(move, PLAYER_O);
                if (gameActive) {
                    currentPlayer = PLAYER_X;
                    updateStatus();
                }
            }, 1000); // 1 секунда
        } else {
            aiMoveInfo.textContent = '';
        }
    }

    function checkWinner(currentBoard) {
        for (const pattern of winPatterns) {
            const [a, b, c] = pattern;
            if (currentBoard[a] && currentBoard[a] === currentBoard[b] && currentBoard[a] === currentBoard[c]) {
                return currentBoard[a];
            }
        }
        return null;
    }

    function updateStatus() {
        if (gameActive) {
            statusElement.textContent = (currentPlayer === PLAYER_X) ? 'Ваш ход (X)' : 'Ход ИИ (O)';
        }
    }

    function resetGame() {
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        currentPlayer = PLAYER_X;
        gameActive = true;
        cells.forEach(cell => {
            cell.textContent = '';
            cell.classList.remove('x', 'o');
        });
        updateStatus();
        if (nnVisualizationContainer.querySelector('svg')) {
            updateVisualization(board, Array(9).fill(0));
        }
    }

    cells.forEach(cell => cell.addEventListener('click', handleCellClick));
    resetButton.addEventListener('click', resetGame);

    trainAI();
});