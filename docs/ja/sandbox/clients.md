---
search:
  exclude: true
---
# サンドボックスクライアント

このページでは、サンドボックスでの処理を実行する場所を選択します。ほとんどの場合、`SandboxAgent` の定義はそのまま使用し、[`SandboxRunConfig`][agents.run_config.SandboxRunConfig] のサンドボックスクライアントとクライアント固有のオプションのみを変更します。

!!! warning "ベータ機能"

    サンドボックスエージェントはベータ版です。一般提供までに API の詳細、デフォルト、サポートされる機能が変更される可能性があります。また、今後さらに高度な機能が追加される予定です。

## 選択ガイド

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 目的 | 最初に使用するもの | 理由 |
| --- | --- | --- |
| macOS または Linux で最速のローカル反復開発 | `UnixLocalSandboxClient` | 追加のインストールが不要で、ローカルファイルシステムを使用した開発が簡単です。 |
| 基本的なコンテナ分離 | `DockerSandboxClient` | 指定したイメージを使用して Docker 内で処理を実行します。 |
| ホスト環境での実行または本番環境相当の分離 | ホスト型サンドボックスクライアント | ワークスペースの境界をプロバイダー管理の環境へ移します。 |

</div>

## ローカルクライアント

ほとんどのユーザーは、次の 2 つのサンドボックスクライアントのいずれかから開始することをお勧めします。

<div class="sandbox-nowrap-first-column-table" markdown="1">

| クライアント | インストール | 適している状況 | コード例 |
| --- | --- | --- | --- |
| `UnixLocalSandboxClient` | なし | macOS または Linux で最速のローカル反復開発を行う場合。ローカル開発に適したデフォルトです。 | [Unix-local スターター](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/unix_local_runner.py) |
| `DockerSandboxClient` | `openai-agents[docker]` | コンテナ分離が必要な場合、またはローカル環境との整合性を保つために特定のイメージを使用する場合。 | [Docker スターター](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/docker/docker_runner.py) |

</div>

Unix-local は、ローカルファイルシステムを対象とした開発を開始する最も簡単な方法です。より強力な環境分離や本番環境相当の整合性が必要になった場合は、Docker またはホスト型プロバイダーへ移行してください。

`SandboxPathGrant.host_path` は Docker 専用であり、ホスト上のパスをコンテナ内の別の POSIX パスへマッピングします。Unix-local では、同一パスへの許可のみがサポートされます。詳細については、[マニフェストのパス許可](guide.md#manifest)を参照してください。

Unix-local から Docker へ切り替えるには、エージェント定義をそのまま維持し、実行設定のみを変更します。

```python
from docker import from_env as docker_from_env

from agents.run import RunConfig
from agents.sandbox import SandboxRunConfig
from agents.sandbox.sandboxes.docker import DockerSandboxClient, DockerSandboxClientOptions

run_config = RunConfig(
    sandbox=SandboxRunConfig(
        client=DockerSandboxClient(docker_from_env()),
        options=DockerSandboxClientOptions(image="python:3.14-slim"),
    ),
)
```

コンテナ分離またはイメージの整合性が必要な場合に使用してください。[examples/sandbox/docker/docker_runner.py](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/docker/docker_runner.py)を参照してください。

## マウントとリモートストレージ

マウントエントリは公開するストレージを記述し、マウント戦略はサンドボックスバックエンドがそのストレージを接続する方法を記述します。組み込みのマウントエントリと汎用戦略は `agents.sandbox.entries` からインポートします。ホスト型プロバイダー向けの戦略は、`agents.extensions.sandbox` またはプロバイダー固有の拡張パッケージから利用できます。

一般的なマウントオプションは次のとおりです。

- `mount_path`: サンドボックス内でストレージが配置される場所です。相対パスはマニフェストのルートを基準に解決され、絶対パスはそのまま使用されます。
- `read_only`: デフォルトは `True` です。サンドボックスからマウント済みストレージへ書き戻す必要がある場合にのみ、`False` に設定してください。
- `mount_strategy`: 必須です。マウントエントリとサンドボックスバックエンドの両方に適合する戦略を使用してください。

マウントは、一時的なワークスペースエントリとして扱われます。スナップショットと永続化のフローでは、マウントされたリモートストレージを保存済みワークスペースへコピーする代わりに、マウント済みパスを切り離すかスキップします。

汎用のローカル／コンテナ戦略は次のとおりです。

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 戦略またはパターン | 適している状況 | 注記 |
| --- | --- | --- |
| `InContainerMountStrategy(pattern=RcloneMountPattern(...))` | サンドボックスイメージで `rclone` を実行できる場合。 | S3、GCS、R2、Azure Blob、Box をサポートします。`RcloneMountPattern` は `fuse` モードまたは `nfs` モードで実行できます。 |
| `InContainerMountStrategy(pattern=MountpointMountPattern(...))` | イメージに `mount-s3` が含まれており、Mountpoint 形式で S3 または S3 互換ストレージへアクセスする場合。 | `S3Mount` と `GCSMount` をサポートします。 |
| `InContainerMountStrategy(pattern=FuseMountPattern(...))` | イメージに `blobfuse2` と FUSE のサポートが含まれている場合。 | `AzureBlobMount` をサポートします。 |
| `InContainerMountStrategy(pattern=S3FilesMountPattern(...))` | イメージに `mount.s3files` が含まれており、既存の S3 Files マウントターゲットへ接続できる場合。 | `S3FilesMount` をサポートします。 |
| `DockerVolumeMountStrategy(driver=...)` | コンテナの起動前に、Docker でボリュームドライバーを使用したマウントを接続する場合。 | Docker 専用です。S3、GCS、R2、Azure Blob、Box は `rclone` をサポートし、S3 と GCS は `mountpoint` もサポートします。 |

</div>

## サポート対象のホスト型プラットフォーム

ホスト型環境が必要な場合でも、通常は同じ `SandboxAgent` 定義をそのまま使用でき、[`SandboxRunConfig`][agents.run_config.SandboxRunConfig] のサンドボックスクライアントのみを変更します。

このリポジトリのチェックアウトではなく公開版 SDK を使用している場合は、対応するパッケージの追加依存関係を通じてサンドボックスクライアントの依存関係をインストールしてください。

プロバイダー固有の設定に関する注記と、リポジトリに含まれる拡張機能のコード例へのリンクについては、[examples/sandbox/extensions/README.md](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/README.md)を参照してください。

<div class="sandbox-nowrap-first-column-table" markdown="1">

| クライアント | インストール | コード例 |
| --- | --- | --- |
| `BlaxelSandboxClient` | `openai-agents[blaxel]` | [Blaxel ランナー](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/blaxel_runner.py) |
| `CloudflareSandboxClient` | `openai-agents[cloudflare]` | [Cloudflare ランナー](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/cloudflare_runner.py) |
| `DaytonaSandboxClient` | `openai-agents[daytona]` | [Daytona ランナー](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/daytona/daytona_runner.py) |
| `E2BSandboxClient` | `openai-agents[e2b]` | [E2B ランナー](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/e2b_runner.py) |
| `ModalSandboxClient` | `openai-agents[modal]` | [Modal ランナー](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/modal_runner.py) |
| `RunloopSandboxClient` | `openai-agents[runloop]` | [Runloop ランナー](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/runloop/runner.py) |
| `VercelSandboxClient` | `openai-agents[vercel]` | [Vercel ランナー](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/vercel_runner.py) |

</div>

ホスト型サンドボックスクライアントは、プロバイダー固有のマウント戦略を提供します。ストレージプロバイダーに最も適したバックエンドとマウント戦略を選択してください。

<div class="sandbox-nowrap-first-column-table" markdown="1">

| バックエンド | マウントに関する注記 |
| --- | --- |
| Docker | `InContainerMountStrategy` や `DockerVolumeMountStrategy` などのローカル戦略を使用して、`S3Mount`、`GCSMount`、`R2Mount`、`AzureBlobMount`、`BoxMount`、`S3FilesMount` をサポートします。 |
| `ModalSandboxClient` | `S3Mount`、`R2Mount`、HMAC 認証済みの `GCSMount` で、`ModalCloudBucketMountStrategy` を使用した Modal クラウドバケットのマウントをサポートします。インライン認証情報または名前付き Modal Secret を使用できます。 |
| `CloudflareSandboxClient` | `S3Mount`、`R2Mount`、HMAC 認証済みの `GCSMount` で、`CloudflareBucketMountStrategy` を使用した Cloudflare バケットのマウントをサポートします。 |
| `BlaxelSandboxClient` | `S3Mount`、`R2Mount`、`GCSMount` で、`BlaxelCloudBucketMountStrategy` を使用したクラウドバケットのマウントをサポートします。また、`agents.extensions.sandbox.blaxel` の `BlaxelDriveMount` と `BlaxelDriveMountStrategy` を使用した永続的な Blaxel Drive もサポートします。 |
| `DaytonaSandboxClient` | `DaytonaCloudBucketMountStrategy` を使用した、rclone ベースのクラウドストレージマウントをサポートします。`S3Mount`、`GCSMount`、`R2Mount`、`AzureBlobMount`、`BoxMount` と組み合わせて使用してください。 |
| `E2BSandboxClient` | `E2BCloudBucketMountStrategy` を使用した、rclone ベースのクラウドストレージマウントをサポートします。`S3Mount`、`GCSMount`、`R2Mount`、`AzureBlobMount`、`BoxMount` と組み合わせて使用してください。 |
| `RunloopSandboxClient` | `RunloopCloudBucketMountStrategy` を使用した、rclone ベースのクラウドストレージマウントをサポートします。`S3Mount`、`GCSMount`、`R2Mount`、`AzureBlobMount`、`BoxMount` と組み合わせて使用してください。 |
| `VercelSandboxClient` | `S3Mount` で `VercelCloudBucketMountStrategy` を使用した、作成時のみの S3 および S3 互換バケットのマウントをサポートします。マウントされたセッションは再開できません。また、インライン認証情報を使用するには `allow_s3_credential_exposure=True` が必要です。 |

</div>

次の表は、各バックエンドが直接マウントできるリモートストレージエントリをまとめたものです。

<div class="sandbox-nowrap-first-column-table" markdown="1">

| バックエンド | AWS S3 | Cloudflare R2 | GCS | Azure Blob Storage | Box | S3 Files |
| --- | --- | --- | --- | --- | --- | --- |
| Docker | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ModalSandboxClient` | ✓ | ✓ | ✓ | - | - | - |
| `CloudflareSandboxClient` | ✓ | ✓ | ✓ | - | - | - |
| `BlaxelSandboxClient` | ✓ | ✓ | ✓ | - | - | - |
| `DaytonaSandboxClient` | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `E2BSandboxClient` | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `RunloopSandboxClient` | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `VercelSandboxClient` | ✓ | - | - | - | - | - |

</div>

実行可能なコード例をさらに確認するには、ローカル、コーディング、メモリ、ハンドオフ、エージェント構成のパターンについては [examples/sandbox/](https://github.com/openai/openai-agents-python/tree/main/examples/sandbox)を、ホスト型サンドボックスクライアントについては [examples/sandbox/extensions/](https://github.com/openai/openai-agents-python/tree/main/examples/sandbox/extensions)を参照してください。