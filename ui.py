import os
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

def render_background():
    html_code = """
    <html>
        <head>
            <style>
                body, html {
                    margin: 0;
                    padding: 0;
                    height: 100%;
                    overflow: hidden;
                }

                /* 背景画像のスタイル */
                .background {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: url("https://miosync.link/img/Amadeus_stand.png");
                    background-size: cover;
                    background-position: center center;
                    background-repeat: no-repeat;
                    z-index: 1;  /* アニメーションの下に配置 */
                }

                /* マトリックス風アニメーションを上に表示 */
                canvas {
                    position: absolute;
                    top: 0;
                    left: 0;
                    z-index: -1;  /* 背景の上に配置 */
                }
            </style>
        </head>
        <body>
            <!-- 背景画像 -->
            <div class="background"></div>

            <!-- 背景のCanvas（マトリックス風アニメーション） -->
            <canvas id="backgroundCanvas"></canvas>

            <script>
                const canvas = document.getElementById('backgroundCanvas');
                const context = canvas.getContext('2d');

                // 画面サイズに合わせてキャンバスのサイズを設定
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;

                const katakana = 'ABCDEFGHIJKLMNOPQRSTUVWXYZウAmadeus12345678ニモロケセユヱ12345678エツヰ12345ノミナカヌシタカミム12345678ミスタマ'.split('');
                const latin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
                const nums = '0123456789'.split('');

                const alphabet = katakana.concat(latin).concat(nums);

                const fontSize = 16;
                const columns = canvas.width / fontSize;

                const rainDrops = [];

                for (let x = 0; x < columns; x++) {
                    rainDrops[x] = 1;
                }

                const draw = () => {
                    context.fillStyle = 'rgba(0, 0, 0, 0.05)';
                    context.fillRect(0, 0, canvas.width, canvas.height);

                    context.fillStyle = '#0F0';
                    context.font = fontSize + 'px monospace';

                    for (let i = 0; i < rainDrops.length; i++) {
                        const text = alphabet[Math.floor(Math.random() * alphabet.length)];
                        context.fillText(text, i * fontSize, rainDrops[i] * fontSize);

                        if (rainDrops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                            rainDrops[i] = 0;
                        }
                        rainDrops[i]++;
                    }
                };

                setInterval(draw, 30);

                // 画面サイズ変更時にキャンバスサイズを再調整
                window.addEventListener('resize', () => {
                    canvas.width = window.innerWidth;
                    canvas.height = window.innerHeight;
                });
            </script>
        </body>
    </html>
    """
    components.html(html_code)  # height オプションは削除
