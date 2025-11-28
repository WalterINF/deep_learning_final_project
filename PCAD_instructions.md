
# Laboratório de Processamento Paralelo e Distribuído (LPPD)

## Índice

## Índice[Close](https://pcad.inf.ufrgs.br/#)

*   [1\. Apresentação](https://pcad.inf.ufrgs.br/#orgad57738)
*   [2\. Características técnicas](https://pcad.inf.ufrgs.br/#orge2c1100)
    *   [2.1. Detalhamento técnico](https://pcad.inf.ufrgs.br/#orgdcac903)
    *   [2.2. Armazenamento](https://pcad.inf.ufrgs.br/#org1dd20e9)
*   [3\. Conta, acesso e submissão](https://pcad.inf.ufrgs.br/#conta)
    *   [3.1. Solicitação de Conta no PCAD](https://pcad.inf.ufrgs.br/#orgaaa737f)
    *   [3.2. Acesso e Submissão](https://pcad.inf.ufrgs.br/#org16414df)
*   [4\. Gerenciador de filas](https://pcad.inf.ufrgs.br/#orgec9d1fc)
*   [5\. Submissão de Jobs](https://pcad.inf.ufrgs.br/#org5341f0a)
    *   [5.1. Jobs Não-Interativos (sbatch)](https://pcad.inf.ufrgs.br/#orgdfa73ef)
    *   [5.2. Jobs Interativos (salloc)](https://pcad.inf.ufrgs.br/#org6663525)
    *   [5.3. Jobs com múltiplos nós não-interativos (sbatch)](https://pcad.inf.ufrgs.br/#org13d27c7)
    *   [5.4. Jobs multi-partição não-interativo (sbatch)](https://pcad.inf.ufrgs.br/#org827e74a)
    *   [5.5. Remover jobs da fila ou em execução](https://pcad.inf.ufrgs.br/#org7bd08ad)
*   [6\. Comandos especiais e administrativos](https://pcad.inf.ufrgs.br/#orga66a1fb)
    *   [6.1. Controle de CPU](https://pcad.inf.ufrgs.br/#org4041f70)
    *   [6.2. Configurações do Kernel](https://pcad.inf.ufrgs.br/#org2f1dfe0)
    *   [6.3. Frequência GPU](https://pcad.inf.ufrgs.br/#org3f1311a)
*   [7\. Referenciando o PCAD](https://pcad.inf.ufrgs.br/#referenciando)
    *   [7.1. Citação global](https://pcad.inf.ufrgs.br/#org1b53e5f)
    *   [7.2. Citação de fomento específico](https://pcad.inf.ufrgs.br/#org93068c5)
*   [8\. Equipe e Contato](https://pcad.inf.ufrgs.br/#org2198bec)

## 1. Apresentação

O [LPPD - Laboratório de Processamento Paralelo e Distribuído](https://pnipe.mcti.gov.br/laboratory/14409), parte do [Grupo de Processamento Paralelo e Distribuído](https://www.inf.ufrgs.br/gppd/) e vinculado ao [Instituto de Informática da UFRGS](https://www.inf.ufrgs.br/), mantém o Parque Computacional de Alto Desempenho (PCAD), uma infraestrutura voltada principalmente à pesquisa, desenvolvimento, ensino e extensão da comunidade científica. Trata-se de uma infraestrutura computacional utilizada por inúmeros grupos de pesquisa em computação e áreas correlatadas. Ele consiste em um conjunto de nós computacionais (servidores) interligados e que podem ser utilizados conjuntamente para a execução de aplicações paralelas de maior porte. O acesso aos recursos é realizado de forma remota por meio de um portal que organiza as solicitações de uso com base nos tipos de projetos aos quais os usuários estão vinculados. Como medida de segurança, nenhum usuário possui acesso físico à sala do Data Center onde os equipamentos estão instalados, no próprio Instituto de Informática.

## 2. Características técnicas

O PCAD possui um nó computacional físico que serve como portal, que pode ser visto como o front-end que centraliza os logins de usuários e permite a alocação de recursos. No front-end estão centralizados também o /home de cada usuário. Os recursos que podem ser alocados a partir do front-end são representados por um conjunto de nós computacionais para execução de aplicações paralelas. Cada nó computacional possui armazenamento local temporário. Estas características são detalhadas a seguir.

### 2.1. Detalhamento técnico

O PCAD possui mais de 1.000 núcleos de CPU e 100.000 de GPU (CUDA threads), distribuídos em mais de 40 nós computacionais. Pode-se compreender um nó como sendo um conjunto de processamento com configurações específicas. Por exemplo, cada nó possui uma determinada quantidade de um processador específico, uma quantidade específica de memória RAM por processador, uma GPU específica, agrupados no que chamamos de rack. Desse modo, temos diferentes nós de computação que apresentam diferentes características e configurações. O usuário pode escolher entre os diversos nós de acordo com sua necessidade específica. Por exemplo, para executar uma aplicação de IA, o usuário escolheria o nó hype, devido à presença de uma GPU NVIDIA Tesla K80, enquanto que para uma aplicação que execute exclusivamente em CPU, pode-se fazer uso de um nó sem GPU.

As partições representam a forma como esses nós estão organizados. Como pode-se ter mais de um nó com as mesmas características, dizemos que os nós que possuem as mesmas características fazem parte de uma partição. Como exemplo temos os 7 nós draco, que são 7 máquinas com as mesmas características agrupados sob a partição draco.

Os nós existentes no PCAD são detalhados a seguir:

      
| Nome | Partição | CPU | RAM | Acelerador | Disco | Placa-mãe |
| --- | --- | --- | --- | --- | --- | --- |
| Nome | Partição | CPU | RAM | Acelerador | Disco | Placa-mãe |
| --- | --- | --- | --- | --- | --- | --- |
| gppd-hpc | \-  | 2 x Intel(R) Xeon(R) E5-2630, 2.30 GHz, 24 threads, 12 cores | 32 GB DDR3 |     | 73.4 TB HDD | Dell Inc. 0H5J4J |
| bali2 | bali | 2 x Intel(R) Xeon(R) E5-2650, 2.00 GHz, 32 threads, 16 cores | 32 GB DDR3 |     | 931.5 GB HDD | Silicon Graphics International X9DRG-HF |
| beagle | beagle | 2 x Intel(R) Xeon(R) E5-2650, 2.00 GHz, 32 threads, 16 cores | 32 GB DDR3 | 2 x NVIDIA GeForce GTX 1080 Ti | 931 GB HDD | Dell Inc. |
| blaise | blaise | 2 x Intel(R) Xeon(R) E5-2699 v4, 2.20 GHz, 88 threads, 44 cores | 256 GB DDR4 |     | 1.8 TB SSD, 1.8 TB HDD | Supermicro X10DGQ |
| cei4 | cei | 2 x Intel(R) Xeon(R) Silver 4116, 2.10 GHz, 48 threads, 24 cores | 96 GB DDR4 |     | 21.8 TB HDD, 894.3 GB SSD | Supermicro X11DPU |
| cei\[1,2,3,5,6\] | cei | 2 x Intel(R) Xeon(R) Silver 4116, 2.10 GHz, 48 threads, 24 cores | 96 GB DDR4 |     | 894.3 GB SSD, 21.8 TB HDD | Supermicro X11DPU |
| cidia | cidia | 2 x Intel(R) Xeon(R) Silver 4208, 2.10 GHz, 32 threads, 16 cores | 320 GB DDR4 | 2 x NVIDIA GeForce RTX 2080 Ti | 7.3 TB HDD, 1.7 TB SSD | Supermicro X11DAi-N |
| draco7 | draco | 2 x Intel(R) Xeon(R) E5-2640 v2, 2.00 GHz, 32 threads, 16 cores | 128 GB DDR3 | 2 x NVIDIA Tesla K20m | 931 GB HDD | Dell Inc. 0KR8W3 |
| draco\[1,2,5,6\] | draco | 2 x Intel(R) Xeon(R) E5-2640 v2, 2.00 GHz, 32 threads, 16 cores | 64 GB DDR3 | NVIDIA Tesla K20m | 1.8 TB HDD | Dell Inc. 0KR8W3 |
| draco\[3,4\] | draco | 2 x Intel(R) Xeon(R) E5-2640 v2, 2.00 GHz, 32 threads, 16 cores | 64 GB DDR3 | NVIDIA Tesla K20m | 931 GB HDD | Dell Inc. 0KR8W3 |
| grace2 | grace | 2 x Grace A02, 3.40 GHz, 144 threads, 144 cores | 480 GB LPDDR5 |     | 894.3 GB NVME, 3.5 TB NVME | Supermicro G1SMH |
| grace1 | grace | 2 x Grace A02, 3.43 GHz, 144 threads, 144 cores | 480 GB LPDDR5 |     | 894.3 GB NVME, 3.5 TB NVME | Supermicro G1SMH |
| hype1 | hype | 2 x Intel(R) Xeon(R) E5-2650 v3, 2.30 GHz, 40 threads, 20 cores | 128 GB DDR4 |     | 558.9 GB HDD | HP ProLiant DL380 Gen9 |
| hype\[2,3\] | hype | 2 x Intel(R) Xeon(R) E5-2650 v3, 2.30 GHz, 40 threads, 20 cores | 128 GB DDR4 |     | 558.9 GB HDD | HP ProLiant XL170r Gen9 |
| hype5 | hype | 2 x Intel(R) Xeon(R) E5-2650 v3, 2.30 GHz, 40 threads, 20 cores | 128 GB DDR4 | 2 x NVIDIA Tesla K80 | 558.9 GB HDD | HP ProLiant XL190r Gen9 |
| knl\[1,2,3,4\] | knl | Intel(R) Xeon Phi(TM) 7250, 1.40 GHz, 68 threads, 68 cores | 112 GB DDR4 |     | 447.1 GB SSD | Intel Corporation S7200AP |
| lunaris | lunaris | AMD Ryzen Threadripper PRO 5965WX 24-Cores, 3.80 GHz, 48 threads, 24 cores | 256 GB DDR4 | 2 x AMD Radeon RX 7900 XT/7900 XTX | 1.8 TB NVME | ASUSTeK COMPUTER INC. Pro WS WRX80E-SAGE SE WIFI |
| marcs | marcs | Intel(R) Core(TM) i7-10700F, 2.87 GHz, 16 threads, 8 cores | 64 GB DDR4 |     | 1.8 TB HDD, 232.9 GB NVME | ASUSTeK COMPUTER INC. TUF GAMING B460M-PLUS |
| phoenix | phoenix | 2 x Intel(R) Xeon(R) Gold 5317, 3.00 GHz, 48 threads, 24 cores | 128 GB DDR4 |     | 223.6 GB SSD | Supermicro X12DPL-NT6 |
| poti\[1,2,3,4,5\] | poti | Intel(R) Core(TM) i7-14700KF, 3.40 GHz, 28 threads, 20 cores | 96 GB DDR5 | NVIDIA GeForce RTX 4070 | 1.7 TB SSD, 119.2 GB NVME | Gigabyte Technology Co., Ltd. Z790 UD AX |
| saude | saude | Intel(R) Xeon(R) E5-2620 v4, 2.10 GHz, 16 threads, 8 cores | 131 GB DDR4 | 3 x NVIDIA GeForce GTX 1080 Ti | 7.3 TB HDD | ASUSTeK COMPUTER INC. X99-E-10G WS |
| sirius | sirius | AMD Ryzen 9 3950X 16-Core Processor, 3.50 GHz, 32 threads, 16 cores | 64 GB DDR4 | AMD Radeon RX 7900 XT/7900 XTX | 894.3 GB SSD, 1.8 TB HDD, 223.6 GB NVME | Gigabyte Technology Co., Ltd. X570 AORUS PRO |
| tsubasa | tsubasa | 2 x Intel(R) Xeon(R) Gold 6226, 2.70 GHz, 48 threads, 24 cores | 196 GB DDR4 |     | 1.7 TB SSD | Supermicro X11DGQ |
| tupi\[5,6\] | tupi | Intel(R) Core(TM) i9-14900KF, 3.20 GHz, 32 threads, 24 cores | 128 GB DDR5 | NVIDIA GeForce RTX 4090 | 1.7 TB SSD, 1.8 TB NVME | Gigabyte Technology Co., Ltd. Z790 UD AX |
| tupi\[3,4\] | tupi | Intel(R) Core(TM) i9-14900KF, 3.20 GHz, 32 threads, 24 cores | 128 GB DDR5 | NVIDIA GeForce RTX 4090 | 1.8 TB NVME | Gigabyte Technology Co., Ltd. Z790 UD AX |
| tupi2 | tupi | Intel(R) Xeon(R) E5-2620 v4, 2.10 GHz, 16 threads, 8 cores | 224 GB DDR4 | NVIDIA GeForce RTX 4090 | 3.6 TB HDD, 894.3 GB SSD, 223.6 GB SSD | ASUSTeK COMPUTER INC. X99-DELUXE II |
| tupi1 | tupi | Intel(R) Xeon(R) E5-2620 v4, 2.10 GHz, 16 threads, 8 cores | 256 GB DDR4 | NVIDIA GeForce RTX 4090 | 447.1 GB SSD, 1.8 TB SSD | ASUSTeK COMPUTER INC. X99-A II |
| turing | turing | 4 x Intel(R) Xeon(R) X7550, 2.00 GHz, 64 threads, 32 cores | 128 GB DDR3 |     | 4.5 TB HDD | Dell Inc. 0JRJM9 |

*   Partições com máquinas de projetos que requerem permissão do coordenador do projeto
    *   apolo, tsubasa, cidia, blaise, saude, marcs, grace

### 2.2. Armazenamento

Os arquivos de cada usuário estão armazenados no front-end e são acessados pelos nós de computação através de Network File System (NFS). De maneira transparente, o NFS monta os dados do usuário na partição que foi alocada e o usuário acessa os dados como se eles estivessem armazenados localmente. Assim, o diretório `$HOME` de cada usuário é acessível a partir de cada nó de computação.

Usuários podem executar aplicações que leem e escrevem dados diretamente na sua `$HOME`, ou seja, no front-end. No entanto, essas operações serão feitas pela rede, o que as torna lentas, visto que o diretório `$HOME` é um Sistema de Arquivos de Rede (NFS - _Network File Sytem_). O ideal, portanto, é utilizar o diretório `$SCRATCH`, que está sempre montado em um disco local do nó computacional.

O `$SCRATCH` é um diretório temporário presente em cada nó de computação e montado no sistema de arquivos próprio do nó. Cada usuário possui um diretório dentro do scratch (`/scratch/USERNAME`). A variável de ambiente `$SCRATCH` é configurada automaticamente no momento do login. Desse modo, para acessar seu diretório scratch, basta acessar `$SCRATCH`:

cd $SCRATCH

Antes da execução da aplicação, o usuário pode copiar os dados do seu `$HOME` no NFS para o seu scratch, acessando as variáveis `$HOME` e `$SCRATCH`:

cp $HOME/<diretório\_origem> $SCRATCH/<diretório\_destino>

No entanto, os dados são mantidos temporariamente no scratch do nó e não podem ser acessados diretamente por outros nós de computação. Desse modo, o usuário precisa copiar os dados para o seu diretório `$HOME` ao final da execução do job:

cp $SCRATCH/<diretório\_origem> $HOME/<diretório\_destino>

IMPORTANTE: os arquivos presentes no scratch são temporários e podem ser removidos a qualquer momento sem aviso prévio aos usuários. Por isso se sugere que, assim que um job termine sua execução, os dados relevantes sejam transferidos para a área de armazenamento do `$HOME`.

SUPER IMPORTANTE: os dados na `$HOME` e no `$SCRATCH` não possuem backup sendo o usuário o responsável por manter seus backups fora do PCAD. Usuários podem copiar seus arquivos de e para o PCAD usando o comando `scp` do ssh, ou ainda o `rsync` sobre ssh, sendo este o recomendado.

Da máquina pessoal para o PCAD (a partir da máquina pessoal):

rsync --verbose --progress --recursive <diretório\_origem> <usuario\_no\_pcad>@gppd-hpc.inf.ufrgs.br:~/<diretório\_destino>

Do PCAD para a máquina pessoal (a partir da máquina pessoal):

rsync --verbose --progress --recursive <usuario\_no\_pcad>@gppd-hpc.inf.ufrgs.br:~/<diretório\_origem> <diretório\_destino>

Do PCAD para a máquina pessoal (a partir do PCAD):

rsync --verbose --progress --recursive <diretório\_origem> <usuario\_na\_sua\_maquina>@<maquina\_pessoal>:~/<diretório\_destino>

## 3. Conta, acesso e submissão

### 3.1. Solicitação de Conta no PCAD

O PCAD é uma infraestrutura experimental, com o propósito de servir de base para a realização de experimentos computacionais de apoio a pesquisa em computação e áreas afins. O formulário cuja URL encontra-se abaixo serve para solicitar a abertura de uma conta nesta plataforma.

*   [Formulário de Solicitação de Conta no PCAD](https://limesurvey.ufrgs.br/index.php?r=survey/index&sid=657778)

Os pedidos recebidos são avaliados conjuntamente com o professor da UFRGS responsável indicado no formulário. Portanto, entre em contato primeiro com este professor antes de solicitar a conta.

### 3.2. Acesso e Submissão

O sistema operacional do PCAD é o Debian, sendo que o mesmo é acessado por meio de SSH através do host `gppd-hpc.inf.ufrgs.br`. O comando para acessá-lo a partir do terminal é o seguinte:

ssh <usuario\_no\_pcad>@gppd-hpc.inf.ufrgs.br

A submissão dos jobs deve ser feita através do gerenciador de recursos e filas Slurm, detalhado a seguir.

## 4. Gerenciador de filas

O gerenciador de filas utilizado é o Slurm. Todos os jobs devem ser submetidos através do Slurm. Utilize os comandos **sinfo** e **sin** para listar informações sobre as filas, partições e nós. Utilize o comando **squeue** para listar informações sobre os jobs e o escalonamento.

*   As filas de execução (partições) do Slurm são:
    *   shared: beagle, bali1, orion\[1-2\], draco\[1-7\]
    *   apolo: apolo
    *   beagle: beagle
    *   blaise: blaise
    *   cei: cei\[1-6\]
    *   cidia: cidia
    *   draco: draco\[1-7\]
    *   greencloud: bali2
    *   hype: hype\[1-5\]
    *   knl: knl\[1-4\]
    *   orion: orion\[1-2\]
    *   saude: saude
    *   sirius: sirius
    *   tsubasa: tsubasa
    *   tupi: tupi\[1-6\]
    *   turing: turing
    *   grace: grace\[1-2\]
    *   poti: poti\[1-5\]
    *   lunaris: lunaris

Na maior parte das filas, o tempo máximo de alocação é de 24h. A fila **shared** permite o uso parcial do nó, compartilhando o uso com outros usuários. Priorize esta fila para realizar testes iniciais nos seus scrips de experimentos. As filas **cei**, **cidia**, **greencloud,** **saude** e **tsubasa** são exclusivas de projetos.

Deve-se ter o cuidado para ajustar corretamente o tempo de duração do job de acordo com as necessidades, para que o nó não fique alocado sem uso ou seja cancelado por ultrapassar o limite de tempo de execução. Priorize o uso de [Jobs Não-Interativos (sbatch)](https://pcad.inf.ufrgs.br/#orgdfa73ef) na plataforma, pois o recurso computacional é liberado no momento que seu experimento terminar.

## 5. Submissão de Jobs

### 5.1. Jobs Não-Interativos (sbatch)

Jobs não interativos são aqueles onde o usuário submete um _script_ que realiza o experimento nas máquinas. O slurm permite esse tipo de script utilizando diretivas em linhas que começam com `#SBATCH`. Todas as linhas que começam com esta diretiva terão seu conteúdo passado diretamente para o slurm no momento da alocação. O exemplo 1 abaixo aloca um nó (`--nodes=1`) para 16 tarefas (`--ntasks=16`) na partição `draco` (`--partition=draco`) por 2 horas (`--time=02:00:00`). Recomenda-se o uso de parâmetros `--output` e `--error` de forma que a saída padrão e de erro sejam direcionadas para arquivos, permitindo o _debug_ do job. No caso de exemplo, esses arquivos serão o nome do job (`%x`, no caso o nome do arquivo que contém o script) e o seu ID (`%j`). As linhas que não contém a diretiva do `sbatch` serão executadas normalmente.

#### 5.1.1. Exemplo 1

No exemplo abaixo, o comando `hostname` é executado.

#!/bin/bash
#SBATCH --job-name=exemplo1
#SBATCH --partition=draco
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --output=%x\_%j.out
#SBATCH --error=%x\_%j.err

hostname

Assumindo que o conteúdo acima está em um arquivo `draco.slurm`, basta submeter o job da seguinte maneira na portal:

sbatch draco.slurm

#### 5.1.2. Exemplo 2

No exemplo abaixo, um aplicação é executada em 7 nós da partição draco.

#!/bin/bash
#SBATCH --job-name=exemplo2
#SBATCH --partition=draco
#SBATCH --nodes=7
#SBATCH --ntasks=224
#SBATCH --time=6:00:00
#SBATCH --output=%x\_%j.out
#SBATCH --error=%x\_%j.err

mpicc exemplo2.c

MACHINEFILE\="nodes.$SLURM\_JOB\_ID"
srun -l hostname | sort -n | awk '{print $2}' > $MACHINEFILE

mpirun --mca btl ^openib --mca btl\_tcp\_if\_include eno1 --bind-to none -np $SLURM\_NTASKS -machinefile $MACHINEFILE ./a.out

Assumindo que o conteúdo acima está em um arquivo `draco_multi.slurm`, basta submeter o job da seguinte maneira na portal:

sbatch draco\_multi.slurm

#### 5.1.3. Exemplo 3

No exemplo abaixo, um aplicação é executada em 2 nós da partição hype.

#!/bin/bash
#SBATCH --job-name=exemplo3
#SBATCH --partition=hype
#SBATCH --nodes=2
#SBATCH --ntasks=80
#SBATCH --time=2:00:00
#SBATCH --output=%x\_%j.out
#SBATCH --error=%x\_%j.err

mpicc exemplo3.c

MACHINEFILE\="nodes.$SLURM\_JOB\_ID"
srun -l hostname | sort -n | awk '{print $2}' > $MACHINEFILE

mpirun --mca oob\_tcp\_if\_include 192.168.30.0/24 --mca btl\_tcp\_if\_include 192.168.30.0/24 --mca btl\_base\_warn\_component\_unused 0 --bind-to none -np $SLURM\_NTASKS -machinefile $MACHINEFILE ./a.out

Assumindo que o conteúdo acima está em um arquivo `hype.slurm`, basta submeter o job da seguinte maneira na portal:

sbatch hype.slurm

### 5.2. Jobs Interativos (salloc)

Evite o uso de jobs interativos para aumentar a disponibilidade dos recursos para a comunidade.

Para submeter jobs interativos, é necessário utilizar o comando **salloc**, solicitando os recursos a serem utilizados. Quando o **salloc** consegue alocar os recursos solicitados para o job, ele informa ao usuário, o qual pode acessar o nó (via ssh), realizar as suas tarefas localmente e executar a aplicação.

#### 5.2.1. Exemplo 1

Solicita a alocação de qualquer nó da partição draco por 5h.

salloc -p draco -J NOME-JOB-EX1 -t 05:00:00

#### 5.2.2. Exemplo 2

Solicita a alocação da draco6 por 36h.

salloc -p draco -w draco6 -J NOME-JOB-EX2 -t 36:00:00

#### 5.2.3. Exemplo 3

Solicita a alocação de dois nós da partição hype por 24h.

salloc -p hype -N 2 -J NOME-JOB-EX3 -t 24:00:00

#### 5.2.4. Exemplo 4

Solicita a alocação de 6 núcleos da partição shared por 1h. O uso do nó é compartilhado com outros usuários, entretanto, os núcleos solicitados são dedicados.

salloc -p shared -c 6 -J NOME-JOB-EX4 -t 1:00:00

### 5.3. Jobs com múltiplos nós não-interativos (sbatch)

Uma forma muito frequente de uso de um cluster de computadores é o emprego de múltiplas máquinas (nós) para executar a mesma aplicação com emprego de MPI. Assumindo um programa MPI simples do tipo "Olá Mundo" com o seguinte conteúdo no arquivo `olamundo.c`:

#include <mpi.h\>
#include <stdio.h\>
#include <unistd.h\>
int main(int argc, char \*\*argv)
{
  int rank;
  char hostname\[256\];
  MPI\_Init(&argc,&argv);
  MPI\_Comm\_rank(MPI\_COMM\_WORLD, &rank);
  gethostname(hostname,255);
  printf("Olá Mundo!  Eu sou \[%d\] executando em \[%s\].\\n", rank, hostname);
  MPI\_Finalize();
  return 0;
}

Compile-o com `mpicc`:

mpicc olamundo.c -o olamundo

Construa um script slurm chamado `multi-no.slurm` com o conteúdo abaixo. Serão alocados 5 nós com 80 tarefas na partição draco por 2 horas. O primeiro comando `srun` criará um arquivo de máquinas para a execução do programa MPI. Cada linha desta arquivo conterá o nome do nó onde será executada a aplicação MPI; como existem múltiplos cores, o nome do nó será repetido várias vezes.

#!/bin/bash
#SBATCH --job-name=exemplo
#SBATCH --partition=draco
#SBATCH --nodes=5
#SBATCH --ntasks=80
#SBATCH --time=2:00:00
#SBATCH --output=%x\_%j.out
#SBATCH --error=%x\_%j.err

MACHINEFILE\="nodes.$SLURM\_JOB\_ID"
srun -l hostname | sort -n | awk '{print $2}' > $MACHINEFILE

mpirun -np $SLURM\_NTASKS \\
       -machinefile $MACHINEFILE \\
       --mca btl ^openib \\
       --mca btl\_tcp\_if\_include eno2 \\
       --bind-to none -np $SLURM\_NTASKS \\
       ./olamundo

Assumindo que o programa `olamundo` foi compilado na $HOME do usuário e que o _script_ abaixo se encontra no mesmo local, basta submeter o job da maneira habitual:

sbatch multi-no.slurm

### 5.4. Jobs multi-partição não-interativo (sbatch)

O slurm permite a criação de jobs multi-partição, para o caso onde o usuário deseja usar nós heterogêneos de partições diferentes. A [documentação do slurm sobre jobs heterogêneos](https://slurm.schedmd.com/heterogeneous_jobs.html) apresenta um detalhamento bastante rico sobre o assunto. Abaixo apresentamos um exemplo simples para a aplicação do tipo "Olá Mundo!", compilada no binário `olamundo`.

Crie um arquivo `multi-particao.slurm` com o conteúdo abaixo. Solicitaremos nós de três partições (5 nós da `hype`, cada um com 20 tarefas; 5 nós da `draco`, cada um com 16 tarefas, e o único nó da partição turing, com 32 tarefas). A diretiva `packjob` deve ser obrigatoriamente empregada para separar os comandos de alocação para cada partição. Como o programa MPI precisa de um arquivo de máquinas de todo o conjunto, incluímos a função shell `gen_machinefile` que imprime as nós alocados de uma partição de acordo com a quantidade de tarefas alocadas naquele nó. Essa função é chamada para cada uma das partições que foram alocadas, e toda a saída é direcionada para o arquivo de máquinas (variável `MACHINEFILE`)

#SBATCH --time=02:00:00
#SBATCH --output=%x\_%j.out
#SBATCH --error=%x\_%j.err
#SBATCH --nodes=5 --partition=hype --ntasks=20
#SBATCH packjob
#SBATCH --nodes=5 --partition=draco --ntasks=16
#SBATCH packjob
#SBATCH --nodes=1 --partition=turing --ntasks=32

\# Função para gerar um arquivo de máquina para uma partição
function gen\_machinefile {
    SLM\_NODES\=$1
    SLM\_TASKS\=$2

    if \[ -z "$SLM\_NODES" \]; then
        return
    fi

    for host in $(scontrol show hostname $SLM\_NODES); do
        for machine in $(seq 1 $SLM\_TASKS); do
            echo $host
        done
    done
}
\# Três partições foram alocadas, portanto três chamadas a gen\_machinefile
MACHINEFILE\="nodes.$SLURM\_JOB\_ID"
gen\_machinefile $SLURM\_JOB\_NODELIST\_PACK\_GROUP\_0 $SLURM\_NPROCS\_PACK\_GROUP\_0 > $MACHINEFILE
gen\_machinefile $SLURM\_JOB\_NODELIST\_PACK\_GROUP\_1 $SLURM\_NPROCS\_PACK\_GROUP\_1 >> $MACHINEFILE
gen\_machinefile $SLURM\_JOB\_NODELIST\_PACK\_GROUP\_2 $SLURM\_NPROCS\_PACK\_GROUP\_2 >> $MACHINEFILE

\# Definir a quantidade de máquinas baseado no arquivo
SLM\_NTASKS\=$(wc -l $MACHINEFILE | awk '{ print $1 }')

\# Executar a aplicação
mpirun -np $SLM\_NTASKS \\
       -machinefile $MACHINEFILE \\
       --mca oob\_tcp\_if\_include 192.168.30.0/24 \\
       --mca btl\_tcp\_if\_include 192.168.30.0/24 \\
       --mca btl\_base\_warn\_component\_unused 0 \\
       --bind-to none \\
       ./olamundo

### 5.5. Remover jobs da fila ou em execução

Pelo número do job

scancel NUMERO-DO-JOB

Todos jobs do usuário

scancel -u NOME-DO-USUARIO

## 6. Comandos especiais e administrativos

Algumas propriedades das máquinas podem ser alteradas pelo usuário sem permissões administrativas, e são resetadas no inicio de cada alocação.

### 6.1. Controle de CPU

A frequência e governor podem ser visualizados e alterados por qualquer usuário utilizando os comandos `cpufreq-info` e `cpufreq-set` respectivamente.

#### 6.1.1. Exemplo 1

No exemplo abaixo, é definido a frequência máxima do hardware para todos os cores de uma máquina, além do governor `performance`.

for i in $(seq 0 $(( $(nproc) - 1 ))); do
    maxf\=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo\_max\_freq)
    cpufreq-set -c $i -g performance -d $maxf -u $maxf
done

O turbobost pode ser alterado editando diretamente o arquivo: `/sys/devices/system/cpu/cpufreq/boost`.

### 6.2. Configurações do Kernel

O Numa balancing pode ser ativado ou desativado utilizando os comandos (com sudo):

sudo /sbin/sysctl kernel.numa\_balancing=1
sudo /sbin/sysctl kernel.numa\_balancing=0

### 6.3. Frequência GPU

A frequência da GPU (MEM, GRAPHICS) pode ser configurada utilizando o comando:

gpu\_control 715 1480

Esta é a saída do comando acima na máquina `blaise`.

Executing command \[nvidia-smi -ac 715,1480\]
Applications clocks set to "(MEM 715, SM 1480)" for GPU 00000000:05:00.0
Applications clocks set to "(MEM 715, SM 1480)" for GPU 00000000:06:00.0
Applications clocks set to "(MEM 715, SM 1480)" for GPU 00000000:84:00.0
Applications clocks set to "(MEM 715, SM 1480)" for GPU 00000000:85:00.0
All done.

## 7. Referenciando o PCAD

### 7.1. Citação global

Todos os trabalhos que apresentarem resultados no qual utilizarem recursos do PCAD, devem fazer menção aos projetos individuais das máquinas ou referenciar o PCAD. Sugerimos a seguinte mensagem:

Alguns experimentos deste trabalho utilizaram os recursos da infraestrutura PCAD, [http://gppd-hpc.inf.ufrgs.br](http://gppd-hpc.inf.ufrgs.br/), no INF/UFRGS.

Em inglês:

Some experiments in this work used the PCAD infrastructure, [http://gppd-hpc.inf.ufrgs.br](http://gppd-hpc.inf.ufrgs.br/), at INF/UFRGS.

### 7.2. Citação de fomento específico

Os projetos individuais estão referenciados na tabela abaixo:

   
| **Nome** | **Órgão de Fomento** | **Número do projeto de Pesquisa** | **Docente** |
| --- | --- | --- | --- |
| **Nome** | **Órgão de Fomento** | **Número do projeto de Pesquisa** | **Docente** |
| --- | --- | --- | --- |
| grace\[1-2\] | FAPERGS | 22/2551-0000390-7 | [Carla Maria Dal Sasso Freitas](https://www.inf.ufrgs.br/~carla/) |
| marcs | FAPERGS Edital PqG 07/2021 | 21/2551-0002052-0 | [Mariana Recamonde Mendoza](https://www.inf.ufrgs.br/~mrmendoza/) |
| cidia | FAPERGS | 20/2551-0000254-3 | [João Luiz Dihl Comba](https://www.inf.ufrgs.br/~comba/) |
| tupi\[3-6\] | Petrobras | TC 5900.0111175.19.9, 2020/00182-5 | [Lucas Mello Schnorr](https://www.inf.ufrgs.br/~schnorr/) |
| lunaris | Petrobras | TC 5900.0111175.19.9, 2020/00182-5 | [Lucas Mello Schnorr](https://www.inf.ufrgs.br/~schnorr/), [Philippe O. A. Navaux](https://www.inf.ufrgs.br/site/docente/philippe-olivier-alexandre-navaux/) |
| tupi\[1-2\] | FAPERGS | 16/2551-0000348-8 | [Lucas Mello Schnorr](https://www.inf.ufrgs.br/~schnorr/) |
| bali2 | FAPERGS | 16/2551-0000488-9 | [Lisandro Zambenedetti Granville](http://www.inf.ufrgs.br/~granville/) |
| blaise | Petrobras | TC 0050.0102253.16.9 | [Philippe O. A. Navaux](https://www.inf.ufrgs.br/site/docente/philippe-olivier-alexandre-navaux/) |
| tsubasa | Petrobras | TC 0050.0102253.16.9 | [Philippe O. A. Navaux](https://www.inf.ufrgs.br/site/docente/philippe-olivier-alexandre-navaux/) |
| draco | INFRA FINEP - UFRGS |     | [Philippe O. A. Navaux](https://www.inf.ufrgs.br/site/docente/philippe-olivier-alexandre-navaux/) |
| cei | SDECT/RS INCUBADORAS | Edital 02/2016 | [Luciana Nedel](https://www.inf.ufrgs.br/~nedel/) |
| saude |     |     | [Viviane P. Moreira](https://www.inf.ufrgs.br/~viviane/) |
| hype | HPE |     | [Philippe O. A. Navaux](https://www.inf.ufrgs.br/site/docente/philippe-olivier-alexandre-navaux/) |

## 8. Equipe e Contato

Coordenador do PCAD: [Prof. Lucas Mello Schnorr](http://www.inf.ufrgs.br/~schnorr/).

Os membros atuantes com contribuições extremamente importantes:

*   Cristiano Alex Künas (Doutorando) \[2022 - \] - [cakunas@inf.ufrgs.br](mailto:cakunas@inf.ufrgs.br)
*   Lucas Nesi (Pós-Doutorando) \[2018 - \] - [lucas.nesi@inf.ufrgs.br](mailto:lucas.nesi@inf.ufrgs.br)

Ex-membros atuantes com contribuições extremamente importantes:

*   Matheus Serpa (Doutorando) \[2018-2022\] - [msserpa@inf.ufrgs.br](mailto:msserpa@inf.ufrgs.br)

Os usuários podem solicitar ajuda para problemas ou apresentar demandas de instalação/configuração através da lista de difusão (lembrando que o cadastro do e-mail do usuário ocorre no momento da criação da sua conta):

**hpc-users-l@inf.ufrgs.br**

Author: Administração do PCAD ([schnorr@inf.ufrgs.br](mailto:schnorr@inf.ufrgs.br))

Date: 2024

[Emacs](https://www.gnu.org/software/emacs/) 28.1 ([Org](https://orgmode.org/) mode 9.7.26)



rsync --verbose --progress --recursive \<seu_usuario>@gppd-hpc.inf.ufrgs.br:~/AraucariaRF-OpenMP/results/profiling/ \./vtune_results/