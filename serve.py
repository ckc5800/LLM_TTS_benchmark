"""
TTS 벤치마크 보고서 HTTP 서버
사용법: python serve.py [포트]
기본 포트: 9100
"""
import http.server
import socket
import sys
import os
import webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9100
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))

def get_local_ips():
    ips = []
    try:
        # 외부 연결 시 사용하는 IP 찾기
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    ips.append("127.0.0.1")
    return ips

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BENCH_DIR, **kwargs)

    def log_message(self, format, *args):
        # 오디오 파일 요청은 로그 제외 (너무 많음)
        if not any(args[0].endswith(ext) for ext in (".wav", ".mp3", ".ogg")):
            super().log_message(format, *args)

os.chdir(BENCH_DIR)
ips = get_local_ips()

print("=" * 55)
print(f"  TTS 벤치마크 보고서 서버 (포트 {PORT})")
print("=" * 55)
for ip in ips:
    print(f"  http://{ip}:{PORT}/html/multilingual_report.html")
print("=" * 55)
print("  종료: Ctrl+C")
print()

with http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료.")
