
# Set color
RED="\033[31;1m"
GREEN="\033[32;1m"
YELLOW="\033[33;1m"
BLUE="\033[34;1m"
CYAN="\033[36;1m"
PLAIN="\033[0m"

# Info messages
DONE="${GREEN}[DONE]${PLAIN}"
ERROR="${RED}[ERROR]${PLAIN}"
WARNING="${YELLOW}[WARNING]${PLAIN}"

# Font Format
BOLD="\033[1m"
UNDERLINE="\033[4m"

# Current folder
CUR_DIR=`pwd`
SUDO_CMD="sudo"

# OS
OSID=$(grep ^ID= /etc/os-release | cut -d= -f2)
OSVER=$(lsb_release -cs)
OSNUM=$(grep -oE  "[0-9.]+" /etc/issue)
DOCKERINFO=$(${SUDO_CMD} docker info | wc -l)
DOCKERINFOLEN=`expr ${DOCKERINFO} + 0`


if [ $DOCKERINFOLEN -gt 30 ]; then
    echo -e "  ${YELLOW}Docker has installed.${PLAIN}"
else
    lowercase_osid=$(echo ${OSID} | tr '[A-Z]' '[a-z]')
    echo -e "  ${YELLOW}Start install docker for ${lowercase_osid} ${PLAIN}"
    DOCKER_INSTALL_CMD="curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun"
    DOCKER_INSTALL_RUN_CMD=""

    case "$lowercase_osid" in
        ubuntu)
            DOCKER_INSTALL_CMD="curl -fsSL https://test.docker.com -o test-docker.sh"
            DOCKER_INSTALL_RUN_CMD="${SUDO_CMD} sh test-docker.sh"
            ;;
        centos)
            DOCKER_INSTALL_CMD="curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun"
            ;;
        debian)
            DOCKER_INSTALL_CMD="curl -fsSL https://get.docker.com -o get-docker.sh"
            DOCKER_INSTALL_RUN_CMD="${SUDO_CMD} sh get-docker.sh"
            ;;
        \"alios\")
            DOCKER_INSTALL_CMD="curl -fsSL https://get.docker.com -o get-docker.sh"
            DOCKER_INSTALL_RUN_CMD="${SUDO_CMD} sh get-docker.sh"
            ;;
        \"alinux\")
            DOCKER_INSTALL_CMD="${SUDO_CMD} yum -y install dnf"
            DOCKER_INSTALL_RUN_CMD="${SUDO_CMD} dnf -y install docker"
            ;;
        *)
            echo -e "  ${RED}$lowercase_osid is not supported.${PLAIN}"
            ;;
    esac

    echo -e "  Get docker installer: ${GREEN}${DOCKER_INSTALL_CMD}${PLAIN}"
    echo -e "  Get docker run: ${GREEN}${DOCKER_INSTALL_RUN_CMD}${PLAIN}"

    $DOCKER_INSTALL_CMD
    if [ ! -z "$DOCKER_INSTALL_RUN_CMD" ]; then
        $DOCKER_INSTALL_RUN_CMD
    fi
    $SUDO_CMD systemctl start docker

    DOCKERINFO=$(${SUDO_CMD} docker info | wc -l)
    DOCKERINFOLEN=`expr ${DOCKERINFO} + 0`
    if [ $DOCKERINFOLEN -gt 30 ]; then
        echo -e "  ${GREEN}Docker install success, start docker server.${PLAIN}"
        $SUDO_CMD systemctl start docker
    else
        echo -e "  ${RED}Docker install failed!${PLAIN}"
        exit 1
    fi
fi