"""Insert an animation into a notebook WITHOUT re-executing the notebook.

Why this exists: a plain `nbconvert --execute` reruns every cell. DPMM takes
58 minutes, BIRCH 6, and the 3D CNN longer still - and any of them could land
on different numbers than the README already reports. So instead we start one
kernel, run ONLY the newly inserted cell, and splice its output back in.

The inserted code cell is just:

    from IPython.display import Image
    Image(filename='../animations/gifs/<name>.gif')

`IPython.display.Image` with a filename embeds the raw bytes base64 under the
`image/gif` MIME key. That is the same output mechanism already carrying the
175 PNG figures in these notebooks, so it renders wherever those render - and
the bytes are identical to the GIF on disk that README links to.

usage: python animations/inject.py <notebook> <gif-name> <anchor-substring> <<'TXT'
       markdown body for the cell that introduces the animation
       TXT
"""
import sys, json, queue, pathlib

import nbformat
from jupyter_client.manager import start_new_kernel

REPO = pathlib.Path(__file__).resolve().parent.parent


def run_one_cell(source, cwd):
    """Start a kernel in `cwd`, execute `source`, return its outputs."""
    km, kc = start_new_kernel(kernel_name='python3', cwd=str(cwd))
    outputs = []
    try:
        msg_id = kc.execute(source)
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=120)
            except queue.Empty:
                raise RuntimeError('kernel timed out on the animation cell')
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
            t, c = msg['msg_type'], msg['content']
            if t == 'status' and c['execution_state'] == 'idle':
                break
            if t in ('display_data', 'execute_result'):
                outputs.append(nbformat.v4.new_output(t, data=c['data'],
                                                      metadata=c.get('metadata', {})))
            elif t == 'stream':
                outputs.append(nbformat.v4.new_output('stream',
                                                      name=c['name'], text=c['text']))
            elif t == 'error':
                raise RuntimeError('\n'.join(c['traceback']))
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    return outputs


def main():
    nb_rel, gif_name, anchor = sys.argv[1], sys.argv[2], sys.argv[3]
    md_body = sys.stdin.read().strip()

    nb_path = REPO / nb_rel
    gif_path = REPO / 'animations' / 'gifs' / f'{gif_name}.gif'
    if not gif_path.exists():
        raise SystemExit(f'missing GIF: {gif_path}')

    nb = nbformat.read(nb_path, as_version=4)

    # Idempotence: strip any previous injection for this GIF so re-running
    # updates in place instead of stacking duplicates. Both inserted cells
    # carry an `animation` metadata tag - matching on the GIF path alone would
    # miss the markdown cell, which does not mention the path, and every rerun
    # would leave an orphaned heading behind.
    nb.cells = [c for c in nb.cells
                if c.get('metadata', {}).get('animation') != gif_name]

    code = (f"from IPython.display import Image\n"
            f"Image(filename='../animations/gifs/{gif_name}.gif')")

    outputs = run_one_cell(code, cwd=nb_path.parent)
    mimes = {k for o in outputs for k in o.get('data', {})}
    if 'image/gif' not in mimes:
        raise SystemExit(f'cell did not emit image/gif, got {mimes or "nothing"}')

    md_cell = nbformat.v4.new_markdown_cell(md_body)
    md_cell['metadata']['animation'] = gif_name
    code_cell = nbformat.v4.new_code_cell(code)
    code_cell['metadata']['animation'] = gif_name
    code_cell['outputs'] = outputs
    code_cell['execution_count'] = None

    # Renumber nothing else; place the pair right after the anchor cell so the
    # animation sits beside the maths it illustrates.
    idx = next((i for i, c in enumerate(nb.cells)
                if anchor in ''.join(c['source'])), None)
    if idx is None:
        raise SystemExit(f'anchor not found in {nb_rel}: {anchor!r}')
    nb.cells[idx + 1:idx + 1] = [md_cell, code_cell]

    nbformat.write(nb, nb_path)
    kb = gif_path.stat().st_size / 1024
    print(f'{nb_rel}: injected {gif_name} ({kb:.0f} KB) after cell {idx}')


if __name__ == '__main__':
    main()
