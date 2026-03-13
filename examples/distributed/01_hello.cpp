#include <iostream>
#include <vix/ai/distributed/Peer.hpp>
#include <vix/ai/distributed/SecureChannel.hpp>
#include <vix/ai/distributed/FederatedTrainer.hpp>
#include <vix/ai/distributed/SyncManager.hpp>

using namespace vix::ai::distributed;

int main()
{
  Peer me("peer-local");
  SecureChannel ch;
  if (ch.handshake("peer-remote"))
    std::cout << "Secure channel established with peer-remote\n";

  FederatedTrainer fl;
  fl.step();
  fl.step();
  std::cout << "FL round: " << fl.current_round() << "\n";

  SyncManager sm;
  sm.add_peer(me);
  sm.add_peer(Peer{"peer-remote"});
  std::cout << "Synced: " << sm.sync() << " peers, total=" << sm.total_synced() << "\n";
}
