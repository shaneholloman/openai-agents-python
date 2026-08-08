---
search:
  exclude: true
---
# サンドボックスクライアント

このページでは、サンドボックスでの処理を実行する場所を選択できます。ほとんどの場合、[`SandboxRunConfig`][agents.run_config.SandboxRunConfig] でサンドボックスクライアントとクライアント固有のオプションのみを変更し、`SandboxAgent` の定義はそのまま使用します。

!!! warning "ベータ機能"

    サンドボックスエージェントはベータ版です。一般提供までに API の詳細、デフォルト、サポートされる機能が変更される可能性があります。また、今後さらに高度な機能が追加される予定です。

## 選択ガイド

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 目的 | 最初の選択肢 | 理由 |
| --- | --- | --- |
| macOS または Linux での最速のローカル反復 | `UnixLocalSandboxClient` | 追加インストールが不要で、ローカルファイルシステムを使用した開発が容易です。 |
| 基本的なコンテナ分離 | `DockerSandboxClient` | 指定したイメージを使用し、Docker 内で処理を実行します。 |
| ホスト実行または本番環境相当の分離 | ホスト型サンドボックスクライアント | ワークスペースの境界をプロバイダー管理の環境へ移します。 |

</div>

## ローカルクライアント

ほとんどのユーザーには、次の 2 つのサンドボックスクライアントのいずれかを最初に使用することをお勧めします。

<div class="sandbox-nowrap-first-column-table" markdown="1">

| クライアント | インストール | 選択する場合 | コード例 |
| --- | --- | --- | --- |
| `UnixLocalSandboxClient` | なし | macOS または Linux で最速のローカル反復を行う場合。ローカル開発の優れたデフォルトです。 | [Unix ローカルのスターター](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/unix_local_runner.py) |
| `DockerSandboxClient` | `openai-agents[docker]` | コンテナ分離が必要な場合や、対象環境をローカルで再現するために特定のイメージを使用する場合。 | [Docker スターター](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/docker/docker_runner.py) |

</div>

Unix ローカルは、ローカルファイルシステムを対象とした開発を始める最も簡単な方法です。より強力な環境分離や本番環境相当の一貫性が必要になった場合は、Docker またはホスト型プロバイダーへ移行してください。

`SandboxPathGrant.host_path` は Docker 専用で、ホスト上のパスをコンテナ内の別の POSIX パスにマッピングします。Unix ローカルでは、同一パスの許可のみがサポートされます。詳細については、[マニフェストのパス許可](guide.md#manifest)を参照してください。

Unix ローカルから Docker に切り替えるには、エージェント定義をそのまま維持し、実行設定のみを変更します。

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

コンテナ分離が必要な場合や、サンドボックスイメージを別の環境で使用されるイメージと一致させる場合に使用します。[examples/sandbox/docker/docker_runner.py](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/docker/docker_runner.py) を参照してください。

## マウントとリモートストレージ

マウントエントリは公開するストレージを記述し、マウント戦略はサンドボックスバックエンドがそのストレージを接続する方法を記述します。組み込みのマウントエントリと汎用戦略は `agents.sandbox.entries` からインポートします。ホスト型プロバイダーの戦略は、`agents.extensions.sandbox` またはプロバイダー固有の拡張パッケージから利用できます。

一般的なマウントオプションは次のとおりです。

- `mount_path`: サンドボックス内でストレージが表示される場所です。相対パスはマニフェストルートを基準に解決され、絶対パスはそのまま使用されます。
- `read_only`: デフォルトは `True` です。サンドボックスからマウント済みストレージへ書き戻す必要がある場合のみ、`False` を設定します。
- `mount_strategy`: 必須です。マウントエントリとサンドボックスバックエンドの両方に適合する戦略を使用してください。

マウントは一時的なワークスペースエントリとして扱われます。スナップショットと永続化のフローでは、マウント済みのリモートストレージを保存対象のワークスペースへコピーするのではなく、マウント済みパスを切り離すかスキップします。

汎用的なローカル／コンテナ戦略は次のとおりです。

<div class="sandbox-nowrap-first-column-table" markdown="1">

| 戦略またはパターン | 使用する場合 | 注記 |
| --- | --- | --- |
| `InContainerMountStrategy(pattern=RcloneMountPattern(...))` | サンドボックスイメージで `rclone` を実行できる場合。 | S3、GCS、R2、Azure Blob、Box をサポートします。`RcloneMountPattern` は `fuse` モードまたは `nfs` モードで実行できます。 |
| `InContainerMountStrategy(pattern=MountpointMountPattern(...))` | イメージに `mount-s3` があり、Mountpoint 形式で S3 または S3 互換ストレージへアクセスする場合。 | `S3Mount` と `GCSMount` をサポートします。 |
| `InContainerMountStrategy(pattern=FuseMountPattern(...))` | イメージに `blobfuse2` があり、FUSE をサポートしている場合。 | `AzureBlobMount` をサポートします。 |
| `InContainerMountStrategy(pattern=S3FilesMountPattern(...))` | イメージに `mount.s3files` があり、既存の S3 Files マウントターゲットへ接続できる場合。 | `S3FilesMount` をサポートします。 |
| `DockerVolumeMountStrategy(driver=...)` | コンテナの起動前に、Docker でボリュームドライバーを利用するマウントを接続する場合。 | Docker 専用です。S3、GCS、R2、Azure Blob、Box は `rclone` を使用してマウントできます。S3 と GCS は `mountpoint` を使用してマウントすることもできます。 |

</div>

## サポート対象のホスト型プラットフォーム

ホスト型環境が必要な場合、通常は同じ `SandboxAgent` の定義をそのまま使用し、[`SandboxRunConfig`][agents.run_config.SandboxRunConfig] でサンドボックスクライアントのみを変更します。

このリポジトリをチェックアウトしたものではなく、公開版 SDK を使用している場合は、対応するパッケージの extra を使用してサンドボックスクライアントの依存関係をインストールしてください。

プロバイダー固有のセットアップに関する注記と、リポジトリに含まれる拡張機能のコード例へのリンクについては、[examples/sandbox/extensions/README.md](https://github.com/openai/openai-agents-python/blob/main/examples/sandbox/extensions/README.md) を参照してください。

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

ホスト型サンドボックスクライアントは、プロバイダー固有のマウント戦略を公開します。ストレージプロバイダーに最適なバックエンドとマウント戦略を選択してください。

<div class="sandbox-nowrap-first-column-table" markdown="1">

| バックエンド | マウントに関する注記 |
| --- | --- |
| Docker | `InContainerMountStrategy` や `DockerVolumeMountStrategy` などのローカル戦略で、`S3Mount`、`GCSMount`、`R2Mount`、`AzureBlobMount`、`BoxMount`、`S3FilesMount` をサポートします。 |
| `ModalSandboxClient` | `ModalCloudBucketMountStrategy` を `S3Mount`、`R2Mount`、HMAC 認証の `GCSMount` とともに使用することで、クラウドバケットのマウントをサポートします。インライン認証情報または名前付き Modal Secret を使用できます。 |
| `CloudflareSandboxClient` | `CloudflareBucketMountStrategy` を `S3Mount`、`R2Mount`、HMAC 認証の `GCSMount` とともに使用することで、バケットのマウントをサポートします。 |
| `BlaxelSandboxClient` | `BlaxelCloudBucketMountStrategy` と `S3Mount`、`R2Mount`、または `GCSMount` のエントリを組み合わせることで、クラウドバケットのマウントをサポートします。また、`BlaxelDriveMount` と `BlaxelDriveMountStrategy` による永続的な Blaxel Drives もサポートします。どちらも `agents.extensions.sandbox.blaxel` から利用できます。 |
| `DaytonaSandboxClient` | `DaytonaCloudBucketMountStrategy` を使用して `rclone` 経由でクラウドストレージをマウントできます。`S3Mount`、`GCSMount`、`R2Mount`、`AzureBlobMount`、`BoxMount` とともに使用してください。 |
| `E2BSandboxClient` | `E2BCloudBucketMountStrategy` を使用して `rclone` 経由でクラウドストレージをマウントできます。`S3Mount`、`GCSMount`、`R2Mount`、`AzureBlobMount`、`BoxMount` とともに使用してください。 |
| `RunloopSandboxClient` | `RunloopCloudBucketMountStrategy` を使用して `rclone` 経由でクラウドストレージをマウントできます。`S3Mount`、`GCSMount`、`R2Mount`、`AzureBlobMount`、`BoxMount` とともに使用してください。 |
| `VercelSandboxClient` | `VercelCloudBucketMountStrategy` と `S3Mount` のエントリを組み合わせることで、作成時に限り S3 および S3 互換バケットのマウントをサポートします。マウント済みセッションは再開できません。また、インライン認証情報には `allow_s3_credential_exposure=True` が必要です。 |

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

その他の実行可能なコード例については、ローカル、コーディング、メモリ、ハンドオフ、エージェント構成のパターンを扱う [examples/sandbox/](https://github.com/openai/openai-agents-python/tree/main/examples/sandbox) と、ホスト型サンドボックスクライアントを扱う [examples/sandbox/extensions/](https://github.com/openai/openai-agents-python/tree/main/examples/sandbox/extensions) を参照してください。