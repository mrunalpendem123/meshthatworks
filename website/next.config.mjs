/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/install',
        destination:
          'https://raw.githubusercontent.com/mrunalpendem123/meshthatworks/master/scripts/bootstrap.sh',
      },
    ];
  },
  async redirects() {
    return [
      {
        source: '/github',
        destination: 'https://github.com/mrunalpendem123/meshthatworks',
        permanent: false,
      },
      {
        source: '/repo',
        destination: 'https://github.com/mrunalpendem123/meshthatworks',
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
