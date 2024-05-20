import config.config as config
import argparse
import os

from search.BWAS import batchedWeightedAStarSearch
from environment.getEnvironment import getEnvironment
from networks.getNetwork import getNetwork

import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--network", required=True, help="Path of Saved Network", type=str
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path of Config File", type=str
    )
    parser.add_argument(
        "-s", "--scrambleDepth", default=1000, help="Depth of Scramble", type=int
    )
    parser.add_argument(
        "-hf", "--heuristicFunction", default="net", help="net or manhattan", type=str
    )

    parser.add_argument(
        "-ns", "--numSolve", default=1, help="Number to Solve", type=int
    )

    args = parser.parse_args()

    conf = config.Config(args.config)

    env = getEnvironment(conf.puzzle)(conf.puzzleSize)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.heuristicFunction == "net":
        net = getNetwork(conf.puzzle, conf.networkType)(conf.puzzleSize)

        loadPath = args.network
        if not os.path.isfile(loadPath):
            raise ValueError("No Network Saved in this Path")

        net.to(device)
        net.load_state_dict(torch.load(loadPath))  # ["net_state_dict"])
        net.eval()

        heuristicFn = net
    elif args.heuristicFunction == "manhattan" and conf.puzzle == "puzzleN":
        heuristicFn = env.manhattanDistance
    else:
        raise ValueError("Invalid Heuristic Function")

    movesList = []
    numNodesGeneratedList = []
    searchItrList = []
    isSolvedList = []
    timeList = []

    numToSolve = args.numSolve

    for i in range(1, numToSolve + 1):

        # scramble = env.generateScramble(args.scrambleDepth)
        # print("SCRAMBLE: ", scramble)
        # inp = [int(x) for x in input("Enter Scramble: ").split()]
        print("ORDER: UP, FRONT, LEFT, DOWN, BACK, RIGHT")
        inp = input("Enter Scramble: ").split()
        for i in range(len(inp)):
            letter = inp[i]
            if letter == "w":
                inp[i] = 0
            if letter == "g":
                inp[i] = 1
            if letter == "o":
                inp[i] = 2
            if letter == "y":
                inp[i] = 3
            if letter == "b":
                inp[i] = 4
            if letter == "r":
                inp[i] = 5
        
        scramble = torch.tensor(inp, dtype=torch.uint8)
        print(inp)


        (
            moves,
            numNodesGenerated,
            searchItr,
            isSolved,
            time,
        ) = batchedWeightedAStarSearch(
            scramble,
            conf.depthWeight,
            conf.numParallel,
            env,
            heuristicFn,
            device,
            conf.maxSearchItr,
        )

        if moves:
            movesList.append(len(moves))

        numNodesGeneratedList.append(numNodesGenerated)
        searchItrList.append(searchItr)
        isSolvedList.append(isSolved)
        timeList.append(time)

        if isSolved:
            print("Solved!")
            print(scramble)
            print("Moves are %s" % "".join(moves))
            print("Solve Length is %i" % len(moves))
            print("%i Nodes were generated" % numNodesGenerated)
            print("There were %i search iterations" % searchItr)
            print("Time of Solve is %.3f seconds" % time)
        else:
            print(scramble)
            print("Max Search Iterations Reached")
            print("%i Nodes were generated" % numNodesGenerated)
            print("Search time was %.3f seconds" % time)

        print("%d out of %d" % (i, numToSolve), flush=True)

    print("Average Move Count %.2f" % (sum(movesList) / len(movesList)))
    print(
        "Average Nodes Generated: %.2f"
        % (sum(numNodesGeneratedList) / len(numNodesGeneratedList))
    )
    print("Number Solved: %d" % isSolvedList.count(True))
    print("Average Time: %.2f seconds" % (sum(timeList) / len(timeList)))

    print(
        "Average Time of Successful Solves is %.2f seconds"
        % (
            sum([i for (i, v) in zip(timeList, isSolvedList) if v])
            / len([i for (i, v) in zip(timeList, isSolvedList) if v])
        )
    )
    print(
        "Average Search Iterations of Successful Solves is %.2f"
        % (
            sum([i for (i, v) in zip(searchItrList, isSolvedList) if v])
            / len([i for (i, v) in zip(searchItrList, isSolvedList) if v])
        )
    )
    print(
        "Max Search Iterations of Successful Solve is %d"
        % (max([i for (i, v) in zip(searchItrList, isSolvedList) if v]))
    )
