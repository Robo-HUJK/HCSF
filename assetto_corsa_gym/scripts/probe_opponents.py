"""
对手数据探针 - 阶段 C
用途：在 AC 开 Race 模式后，验证能拉到对手数据。
通道：MGMT 2347 端口（TCP，请求-响应），命令 "get_opponents"
返回：JSON 数组，每元素含 id / world_position / speedKMH / yaw / brakeStatus
"""
import json
import socket
import sys
import time

HOST = "127.0.0.1"
PORT = 2347          # MGMT 通道（OPP 2346 通道在 Wine 下 bind 失败，弃用）
MAX_MSG_SIZE = 2**20
N_PROBES = 5


def get_opponents():
    """单次拉取对手快照（请求-响应模式）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        s.connect((HOST, PORT))
        s.sendall(b"get_opponents")
        s.settimeout(5)
        data = s.recv(MAX_MSG_SIZE)
        return json.loads(data.decode("utf8"))


def main():
    print(f"[probe] 连接 MGMT 通道 {HOST}:{PORT}")
    try:
        # 先做一次空连接验证服务端可达
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            s.connect((HOST, PORT))
    except (ConnectionRefusedError, OSError) as e:
        print(f"[probe] ❌ 连接 2347 失败: {e}")
        print("[probe] 检查 AC 是否在跑、插件是否加载（看 ss -ltnp | grep 2347）")
        sys.exit(1)

    print("[probe] ✅ MGMT 通道可达，开始抓 {} 次对手快照...".format(N_PROBES))
    for i in range(N_PROBES):
        try:
            opps = get_opponents()
        except Exception as e:
            print(f"[probe] 第 {i} 次拉取失败: {e}")
            time.sleep(0.5)
            continue
        print(f"\n[probe] === 快照 {i} ===")
        print(f"  对手数: {len(opps)}")
        if not opps:
            print("  ⚠️  对手列表为空 —— 检查比赛是否为 Race 模式且 grid 上有 AI")
            continue
        for j, opp in enumerate(opps):
            keys = sorted(opp.keys())
            print(f"  opp[{j}] 字段: {keys}")
            print(f"    id={opp.get('id')}")
            print(f"    world_position={opp.get('world_position')}")
            print(f"    speedKMH={opp.get('speedKMH', 0):.2f}")
            print(f"    yaw={opp.get('yaw', 0):.4f}")
            brake = opp.get('brakeStatus', None)
            if brake is None:
                print("    brakeStatus=MISSING ❌ (插件未重启或字段未生效)")
            else:
                print(f"    brakeStatus={brake:.3f} ✅")
        time.sleep(0.3)

    print("\n[probe] 完成。如所有字段齐全且 speedKMH/world_position 在变 → ✅ 通路 OK，可进 D 阶段")


if __name__ == "__main__":
    main()
