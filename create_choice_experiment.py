#!/usr/bin/env python3

import argparse
import json
import os
import random
import shutil
from html import escape
from pathlib import Path


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        [
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.name.endswith('_fake.png')
        ]
    )


def experiment_name_from_images_dir(images_dir: Path) -> str:
    if images_dir.name == 'images' and images_dir.parent.name == 'test_latest':
        return images_dir.parent.parent.name
    if images_dir.name == 'images':
        return images_dir.parent.name
    return images_dir.name


def sanitize_name(name: str) -> str:
    cleaned = name.strip().replace('/', '_').replace('\\', '_')
    return cleaned or 'choice_experiment'


def build_pairs(left_images: list[Path], right_images: list[Path], max_pairs: int, rng: random.Random) -> list[dict]:
    pair_count = min(len(left_images), len(right_images))
    if max_pairs > 0:
        pair_count = min(pair_count, max_pairs)

    selected_left = left_images[:pair_count]
    selected_right = right_images[:pair_count]

    pairs = []
    for index, (left_image, right_image) in enumerate(zip(selected_left, selected_right)):
        ordered = [
            {
                'source_label': 'left',
                'source_name': left_image.name,
                'copied_name': f'{index:04d}_left_{left_image.name}',
            },
            {
                'source_label': 'right',
                'source_name': right_image.name,
                'copied_name': f'{index:04d}_right_{right_image.name}',
            },
        ]
        rng.shuffle(ordered)
        pairs.append({'index': index, 'items': ordered})

    rng.shuffle(pairs)
    return pairs


def copy_images(pairs: list[dict], output_images_dir: Path, left_images: list[Path], right_images: list[Path]) -> None:
    output_images_dir.mkdir(parents=True, exist_ok=True)

    copy_map = {}
    for pair in pairs:
        for item in pair['items']:
            copy_map[item['copied_name']] = item

    for copied_name, item in copy_map.items():
        source_name = item['source_name']
        source_path = None
        if item['source_label'] == 'left':
            for path in left_images:
                if path.name == source_name:
                    source_path = path
                    break
        else:
            for path in right_images:
                if path.name == source_name:
                    source_path = path
                    break

        if source_path is None:
            raise FileNotFoundError(f'找不到源图片：{source_name}')

        shutil.copy2(source_path, output_images_dir / copied_name)


def render_html(experiment_name: str, left_name: str, right_name: str, pairs: list[dict]) -> str:
    pairs_json = json.dumps(pairs, ensure_ascii=False)
    experiment_name_json = json.dumps(experiment_name, ensure_ascii=False)
    left_name_json = json.dumps(left_name, ensure_ascii=False)
    right_name_json = json.dumps(right_name, ensure_ascii=False)
    total_pairs = len(pairs)

    return f'''<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{escape(experiment_name)}</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f4f6fb;
            --panel: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --border: #d8dee9;
            --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            --accent: #0f766e;
            --overlay: rgba(120, 120, 120, 0.68);
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: radial-gradient(circle at top, #ffffff 0%, var(--bg) 55%, #eef2f7 100%);
            color: var(--text);
        }}

        .topbar {{
            position: sticky;
            top: 0;
            z-index: 20;
            backdrop-filter: blur(8px);
            background: rgba(244, 246, 251, 0.92);
            border-bottom: 1px solid var(--border);
        }}

        .topbar-inner {{
            max-width: 1440px;
            margin: 0 auto;
            padding: 16px 24px;
            display: flex;
            gap: 16px;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
        }}

        .title-block h1 {{
            margin: 0;
            font-size: 24px;
        }}

        .title-block .desc {{
            margin-top: 6px;
            color: var(--muted);
            font-size: 14px;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(4, minmax(180px, auto));
            gap: 12px;
            align-items: center;
        }}

        .stat-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 10px 14px;
            box-shadow: var(--shadow);
            min-width: 180px;
        }}

        .stat-label {{
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 4px;
        }}

        .stat-value {{
            font-size: 18px;
            font-weight: 700;
        }}

        .container {{
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px;
        }}

        .meta {{
            margin: 0 0 18px;
            color: var(--muted);
            font-size: 14px;
        }}

        .pair-list {{
            display: grid;
            gap: 16px;
        }}

        .pair-row {{
            position: relative;
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }}

        .choice-card {{
            position: relative;
            border-radius: 16px;
            overflow: hidden;
            background: var(--panel);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            min-height: 240px;
            cursor: pointer;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}

        .choice-card:hover {{
            transform: translateY(-1px);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
        }}

        .choice-card img {{
            display: block;
            width: 100%;
            height: auto;
        }}

        .choice-label {{
            padding: 10px 12px 12px;
            border-top: 1px solid var(--border);
            background: #fff;
            font-size: 13px;
            color: #374151;
            word-break: break-all;
        }}

        .choice-hint {{
            position: absolute;
            left: 12px;
            top: 12px;
            z-index: 2;
            background: rgba(17, 24, 39, 0.78);
            color: white;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 12px;
            letter-spacing: 0.02em;
        }}

        .pair {{
            position: relative;
            border-radius: 18px;
        }}

        .pair.locked .choice-card {{
            pointer-events: none;
        }}

        .pair.locked::after {{
            content: '已选择';
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 18px;
            background: var(--overlay);
            color: #fff;
            font-size: 22px;
            font-weight: 700;
            letter-spacing: 0.08em;
            pointer-events: none;
        }}

        .choice-card.selected {{
            outline: 3px solid var(--accent);
            outline-offset: -3px;
        }}

        .footer-note {{
            margin-top: 18px;
            color: var(--muted);
            font-size: 13px;
        }}

        .export-button {{
            width: 100%;
            border: 0;
            border-radius: 12px;
            padding: 12px 14px;
            font-size: 14px;
            font-weight: 700;
            background: linear-gradient(135deg, #0f766e, #14b8a6);
            color: #fff;
            cursor: pointer;
            box-shadow: 0 8px 20px rgba(15, 118, 110, 0.22);
            transition: transform 0.15s ease, opacity 0.15s ease, box-shadow 0.15s ease;
        }}

        .export-button:hover:not(:disabled) {{
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(15, 118, 110, 0.3);
        }}

        .export-button:disabled {{
            cursor: not-allowed;
            opacity: 0.45;
            box-shadow: none;
        }}

        .export-status {{
            margin-top: 8px;
            color: var(--muted);
            font-size: 12px;
            line-height: 1.5;
        }}

        @media (max-width: 980px) {{
            .topbar-inner {{
                align-items: flex-start;
            }}

            .stats {{
                grid-template-columns: 1fr;
                width: 100%;
            }}

            .stat-card {{
                min-width: 0;
            }}
        }}

        @media (max-width: 720px) {{
            .container, .topbar-inner {{
                padding-left: 12px;
                padding-right: 12px;
            }}

            .pair-row {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="topbar">
        <div class="topbar-inner">
            <div class="title-block">
                <h1>{escape(experiment_name)}</h1>
                <div class="desc">强制二选一人类感知实验，点击你认为更真实的一侧。</div>
            </div>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-label">{escape(left_name)} 得分</div>
                    <div class="stat-value" id="count-left">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">{escape(right_name)} 得分</div>
                    <div class="stat-value" id="count-right">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">已完成</div>
                    <div class="stat-value" id="count-done">0 / {total_pairs}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">结果导出</div>
                    <button class="export-button" id="export-button" disabled>完成后导出 result.txt</button>
                    <div class="export-status" id="export-status">先完成全部 {total_pairs} 组选择，按钮才会可用。</div>
                </div>
            </div>
        </div>
    </div>

    <div class="container">
        <p class="meta">共 {total_pairs} 组图片；每组只允许选择一次。选中后该组会自动变灰并锁定。</p>
        <div class="pair-list" id="pair-list"></div>
        <div class="footer-note">刷新页面会重置计数；如果你需要把选择结果保存到文件，我可以再加一个提交/导出按钮。</div>
    </div>

    <script>
        const pairs = {pairs_json};
        const experimentName = {experiment_name_json};
        const leftName = {left_name_json};
        const rightName = {right_name_json};
        const counts = {{ left: 0, right: 0 }};
        const selections = [];
        const pairList = document.getElementById('pair-list');
        const countLeft = document.getElementById('count-left');
        const countRight = document.getElementById('count-right');
        const countDone = document.getElementById('count-done');
        const exportButton = document.getElementById('export-button');
        const exportStatus = document.getElementById('export-status');
        const resultFileName = 'result.txt';
        const dbName = 'choice_experiment_export_db';
        const storeName = 'file_handles';
        const handleKey = 'result_txt_handle';

        function openHandleDatabase() {{
            return new Promise((resolve, reject) => {{
                const request = indexedDB.open(dbName, 1);

                request.onupgradeneeded = () => {{
                    request.result.createObjectStore(storeName);
                }};

                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            }});
        }}

        async function readStoredHandle() {{
            try {{
                const db = await openHandleDatabase();
                return await new Promise((resolve, reject) => {{
                    const transaction = db.transaction(storeName, 'readonly');
                    const store = transaction.objectStore(storeName);
                    const request = store.get(handleKey);
                    request.onsuccess = () => resolve(request.result || null);
                    request.onerror = () => reject(request.error);
                }});
            }} catch (error) {{
                return null;
            }}
        }}

        async function saveStoredHandle(handle) {{
            const db = await openHandleDatabase();
            return await new Promise((resolve, reject) => {{
                const transaction = db.transaction(storeName, 'readwrite');
                const store = transaction.objectStore(storeName);
                const request = store.put(handle, handleKey);
                request.onsuccess = () => resolve(true);
                request.onerror = () => reject(request.error);
            }});
        }}

        async function getWritableHandle() {{
            let handle = await readStoredHandle();

            if (handle) {{
                const permission = await handle.queryPermission({{ mode: 'readwrite' }});
                if (permission === 'granted') {{
                    return handle;
                }}

                const requested = await handle.requestPermission({{ mode: 'readwrite' }});
                if (requested === 'granted') {{
                    return handle;
                }}
            }}

            if (!window.showSaveFilePicker) {{
                throw new Error('当前浏览器不支持直接写入本地文件，请使用 Chromium 内核浏览器并通过 localhost 打开页面。');
            }}

            handle = await window.showSaveFilePicker({{
                suggestedName: resultFileName,
                types: [{{
                    description: 'Text files',
                    accept: {{ 'text/plain': ['.txt'] }},
                }}],
            }});

            await saveStoredHandle(handle);
            return handle;
        }}

        function formatTimestamp(date) {{
            const pad = (value) => String(value).padStart(2, '0');
            return `${{date.getFullYear()}}-${{pad(date.getMonth() + 1)}}-${{pad(date.getDate())}} ${{pad(date.getHours())}}:${{pad(date.getMinutes())}}:${{pad(date.getSeconds())}}`;
        }}

        function buildResultRecord() {{
            const sortedSelections = [...selections].sort((a, b) => a.pair_index - b.pair_index);
            return {{
                timestamp: formatTimestamp(new Date()),
                experiment: experimentName,
                left_model: leftName,
                right_model: rightName,
                total_pairs: pairs.length,
                left_score: counts.left,
                right_score: counts.right,
                left_win_rate: Number((counts.left / pairs.length).toFixed(4)),
                right_win_rate: Number((counts.right / pairs.length).toFixed(4)),
                winner: counts.left === counts.right ? 'tie' : (counts.left > counts.right ? 'left' : 'right'),
                selections: sortedSelections,
            }};
        }}

        async function appendResultLine() {{
            const handle = await getWritableHandle();
            const file = await handle.getFile();
            const existingText = await file.text();
            const record = JSON.stringify(buildResultRecord(), null, 2);
            const separator = existingText && !existingText.endsWith('\\n') ? '\\n' : '';
            const nextText = `${{existingText}}${{separator}}${{record}}\\n`;
            const writable = await handle.createWritable();
            await writable.write(nextText);
            await writable.close();
        }}

        function updateCounters() {{
            countLeft.textContent = counts.left;
            countRight.textContent = counts.right;
            countDone.textContent = `${{counts.left + counts.right}} / ${{pairs.length}}`;
            updateExportButton();
        }}

        function updateExportButton() {{
            const finished = counts.left + counts.right === pairs.length;
            exportButton.disabled = !finished;
            if (finished) {{
                exportButton.textContent = '导出 result.txt';
                exportStatus.textContent = '已完成全部选择，现在可以导出结果。';
            }} else {{
                exportButton.textContent = '完成后导出 result.txt';
                exportStatus.textContent = `先完成全部 ${{pairs.length}} 组选择，按钮才会可用。`;
            }}
        }}

        function lockPair(pairElement, selectedCard) {{
            pairElement.classList.add('locked');
            selectedCard.classList.add('selected');
            pairElement.querySelectorAll('.choice-card').forEach((card) => {{
                card.style.pointerEvents = 'none';
            }});
        }}

        function handleChoice(pairElement, selectedCard) {{
            if (pairElement.dataset.locked === '1') {{
                return;
            }}

            const label = selectedCard.dataset.label;
            selections.push({{
                pair_index: Number(pairElement.dataset.index),
                chosen_label: label,
                chosen_side: selectedCard.dataset.side,
                chosen_image: selectedCard.dataset.sourceName,
            }});
            counts[label] += 1;
            pairElement.dataset.locked = '1';
            lockPair(pairElement, selectedCard);
            updateCounters();
        }}

        function createPairNode(pair, index) {{
            const pairElement = document.createElement('div');
            pairElement.className = 'pair';
            pairElement.dataset.index = String(index);
            pairElement.dataset.locked = '0';

            const rowElement = document.createElement('div');
            rowElement.className = 'pair-row';

            pair.items.forEach((item, position) => {{
                const card = document.createElement('div');
                card.className = 'choice-card';
                card.dataset.label = item.source_label;
                card.dataset.sourceName = item.source_name;
                card.dataset.side = position === 0 ? 'left' : 'right';

                card.innerHTML = `
                    <div class="choice-hint">点击选择</div>
                    <img src="images/${{item.copied_name}}" alt="${{item.source_name}}">
                    <div class="choice-label">${{position === 0 ? '左侧' : '右侧'}} · ${{item.source_name}}</div>
                `;

                card.addEventListener('click', () => handleChoice(pairElement, card));
                rowElement.appendChild(card);
            }});

            pairElement.appendChild(rowElement);
            return pairElement;
        }}

        pairs.forEach((pair, index) => {{
            pairList.appendChild(createPairNode(pair, index));
        }});

        exportButton.addEventListener('click', async () => {{
            if (exportButton.disabled) {{
                return;
            }}

            exportButton.disabled = true;
            exportStatus.textContent = '正在写入 result.txt ...';

            try {{
                await appendResultLine();
                exportStatus.textContent = '已成功写入 result.txt。';
                exportButton.textContent = '已导出';
            }} catch (error) {{
                exportButton.disabled = false;
                exportStatus.textContent = `导出失败：${{error.message}}`;
            }}
        }});

        updateCounters();
    </script>
</body>
</html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='生成一个强制二选一的人类感知实验 HTML 页面。'
    )
    parser.add_argument('--left', required=True, help='左侧类图片所在文件夹路径')
    parser.add_argument('--right', required=True, help='右侧类图片所在文件夹路径')
    parser.add_argument('--name', required=True, help='实验名称，也是 paired_images 下的输出文件夹名')
    parser.add_argument('--output-root', default='/root/autodl-tmp/paired_images', help='输出根目录，默认保存到 paired_images')
    parser.add_argument('--max-pairs', type=int, default=50, help='最多展示多少组，0 表示使用全部')
    parser.add_argument('--seed', type=int, default=None, help='随机种子，不填则使用系统随机')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    left_dir = Path(args.left).expanduser().resolve()
    right_dir = Path(args.right).expanduser().resolve()
    if not left_dir.is_dir():
        raise NotADirectoryError(f'左侧路径不是有效文件夹：{left_dir}')
    if not right_dir.is_dir():
        raise NotADirectoryError(f'右侧路径不是有效文件夹：{right_dir}')

    left_images = collect_images(left_dir)
    right_images = collect_images(right_dir)
    if not left_images:
        raise ValueError(f'左侧文件夹中没有找到 _fake 图片：{left_dir}')
    if not right_images:
        raise ValueError(f'右侧文件夹中没有找到 _fake 图片：{right_dir}')

    if len(left_images) != len(right_images):
        raise ValueError(
            f'两侧 _fake 图片数量不一致：左侧 {len(left_images)} 张，右侧 {len(right_images)} 张'
        )

    safe_name = sanitize_name(args.name)
    output_root = Path(args.output_root).expanduser().resolve()
    output_dir = output_root / safe_name
    output_images_dir = output_dir / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(left_images, right_images, args.max_pairs, rng)
    copy_images(pairs, output_images_dir, left_images, right_images)

    html_content = render_html(
        experiment_name=safe_name,
        left_name=experiment_name_from_images_dir(left_dir),
        right_name=experiment_name_from_images_dir(right_dir),
        pairs=pairs,
    )

    html_path = output_dir / 'index.html'
    with html_path.open('w', encoding='utf-8') as file:
        file.write(html_content)

    manifest_path = output_dir / 'manifest.json'
    with manifest_path.open('w', encoding='utf-8') as file:
        json.dump(
            {
                'left': str(left_dir),
                'right': str(right_dir),
                'output': str(output_dir),
                'pair_count': len(pairs),
                'seed': args.seed,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f'已生成实验页面：{html_path}')
    print(f'图片已复制到：{output_images_dir}')
    print(f'总共生成 {len(pairs)} 组强制二选一图片。')


if __name__ == '__main__':
    main()