# Michael Teixeira
# 1001375188

import sys
import random
import math
import numpy as np

assert len(sys.argv) >= 5, "Too few arguments!"

Classes = []

class Node :
    def __init__(self, attribute, threshold, Gain, ID) :
        self.id = ID
        self.attr = attribute
        self.threshold = threshold
        self.gain = Gain
        self.leftChild = None
        self.rightChild = None

class Leaf :
    def __init__(self, Value, ID) :
        self.id = ID
        self.value = Value

def parceFile(file) :
    classes = []
    data = [line.rstrip('\n').split() for line in open(file)]
    for i in range( len(data) ) :
        data[i] = [float(x) for x in data[i]]

        if ( not( data[i][-1] in classes ) ) :
            classes.append( data[i][-1] )

    classes.sort()
    for i in classes :
        Classes.append(i)
    return data, len(classes)

def distribution(examples, numClasses) :
    classCounts = []

    for i in range(numClasses) :
        classCounts.append(0)

    for i in examples :
        classCounts[ Classes.index( i[-1] ) ] += 1

    return [x / len(examples) for x in classCounts]

def calculateGain(examples, attr, threshold) :
    left = []
    right = []

    baseClasses = []
    baseClassCounts = []
    for x in examples :
        if( x[-1] in baseClasses ) :
            baseClassCounts[ baseClasses.index( x[-1] ) ] += 1
        else :
            baseClasses.append( x[-1] )
            baseClassCounts.append(1)

        if( x[attr] < threshold ) :
            left.append(x)
        else :
            right.append(x)

    entropyBase = 0
    for i in range( len( baseClasses ) ) :
        if( baseClassCounts[i] > 0 ) :
            entropyBase = entropyBase - ( baseClassCounts[i] / len( examples ) ) * math.log( ( baseClassCounts[i] / len( examples ) ), 2 )

    leftClasses = []
    leftClassCounts = []
    for i in left :
        if( i[-1] in leftClasses ) :
            leftClassCounts[ leftClasses.index( i[-1] ) ] += 1
        else :
            leftClasses.append( i[-1] )
            leftClassCounts.append(0)
    
    entropyLeft = 0
    for i in range( len( leftClasses ) ) :
        if( leftClassCounts[i] > 0 ) :
            entropyLeft = entropyLeft - ( leftClassCounts[i] / len( left ) ) * math.log( ( leftClassCounts[i] / len( left ) ), 2 )

    rightClasses = []
    rightClassCounts = []
    for i in right :
        if( i[-1] in rightClasses ) :
            rightClassCounts[ rightClasses.index( i[-1] ) ] += 1
        else :
            rightClasses.append( i[-1] )
            rightClassCounts.append(0)

    entropyRight = 0
    for i in range( len( rightClasses ) ) :
        if( rightClassCounts[i] > 0 ) :
            entropyRight = entropyRight - ( rightClassCounts[i] / len( right ) ) * math.log( ( rightClassCounts[i] / len( right ) ), 2 )
    
    gain = entropyBase - ( len( left ) / len( examples ) ) * entropyLeft - ( len( right ) / len( examples ) ) * entropyRight

    return gain

def choose_attribute(examples, attributes, option) :
    bestAttr = -1
    bestThres = -1
    maxGain = -1
    
    if( option == 'optimized' ) :
        for attr in attributes :
            attributeValues = np.array(examples)[:, attr]
            
            L = min(attributeValues)
            M = max(attributeValues)
            
            for k in range(1, 51) :
                threshold = L + k * ( M - L ) / 51
                gain = calculateGain(examples, attr, threshold)
                
                if( gain > maxGain ) :
                    maxGain = gain
                    bestAttr = attr
                    bestThres = threshold

    elif( option == 'randomized' ) :
        bestAttr = random.choice(attributes)
        attributeValues = np.array(examples)[:, bestAttr]
        L = min(attributeValues)
        M = max(attributeValues)

        for k in range(1, 51) :
            threshold = L + k * ( M - L ) / 51
            gain = calculateGain(examples, bestAttr, threshold)

            if( gain > maxGain ) :
                maxGain = gain
                bestThres = threshold
        
    else :
        print( "Invalid parameter passed to choose_attrubute!\nExiting..." )
        sys.exit()

    return bestAttr, bestThres, maxGain

def all_same(examples) :
    targetClass = examples[0][-1]

    for i in examples :
        if( i[-1] != targetClass ) :
            return False

    return True

def DTL(examples, attributes, default, option, pruningThreshold, nodeID, numClasses) :
    if( len(examples) < int(pruningThreshold) ) :
        return Leaf( default, nodeID + 1 ) # return leaf node
    elif( all_same(examples) ) :
        return Leaf( examples[0][-1], nodeID + 1 ) # return leaf node
    else :
        bestAttr, bestThres, gain = choose_attribute(examples, attributes, option)
        tree = Node( bestAttr, bestThres, gain, nodeID )
        
        examplesLeft = []
        examplesRight = []

        for i in examples :
            if( i[tree.attr] < tree.threshold ) :
                examplesLeft.append(i)
            else :
                examplesRight.append(i)
        
        tree.leftChild = DTL( examplesLeft, attributes, distribution(examples, numClasses), option, pruningThreshold, tree.id * 2, numClasses )
        tree.rightChild = DTL( examplesRight, attributes, distribution(examples, numClasses), option, pruningThreshold, (tree.id * 2) + 1, numClasses )

        return tree

def top_DTL(trainingData, option, pruningThreshold, numClasses) :
    attributes = []
    for i in range( len(trainingData[0][:-1]) ) :
        attributes.append(i)
    
    default = distribution(trainingData, numClasses)
    return DTL(trainingData, attributes, default, option, pruningThreshold, 1, numClasses)

def classifyObject(testObject, trees) :
    results = []
    for tree in trees :
        node = tree
        while( type(node) != Leaf ) :
            if( testObject[node.attr] < node.threshold ) :
                node = node.leftChild
            else :
                node = node.rightChild
        
        results.append(node.value)

    maxValues = []
    maxIndeices = []
    for treeRes in results :
        maxValue = 0
        maxIndex = 0
        for i in range( len( treeRes ) ) :
            if( treeRes[i] > maxValue ) :
                maxValue = treeRes[i]
                maxIndex = i

        maxValues.append(maxValue)
        maxIndeices.append(maxIndex)

    maxCount = maxValues.count( max(maxValues) )
    prediction = maxIndeices[ maxValues.index( max(maxValues) ) ] + 1

    if( testObject[-1] == prediction ) :
        if( maxCount > 1 ) :
            accuracy = 1 / maxCount
        else :
            accuracy = 1
    else :
        accuracy = 0
    
    return prediction, accuracy

def printTree(node, treeId) :
    if( type(node) == Leaf ) :
        print("tree=%2d, node=%3d, feature=N/A, thr=N/A, gain=N/A\n" % ( treeId, node.id ) )
        return
    else :
        print("tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n" % ( treeId, node.id, node.attr, node.threshold, node.gain ) )
        printTree(node.leftChild, treeId)
        printTree(node.rightChild, treeId)
        return

def decision_tree( trainingFile, testFile, option, pruningThreshold ) :
    trainingData, numClasses = parceFile(trainingFile)
    trees = []

    ### Train Phase ###
    if ( option == "optimized" ) :
        trees.append( top_DTL( trainingData, option, pruningThreshold, numClasses ) )

    elif ( option == "randomized" ) :
        trees.append( top_DTL( trainingData, option, pruningThreshold, numClasses ) )

    elif ( option == "forest3" ) :
        for i in range(3) :
            trees.append( top_DTL( trainingData, "randomized", pruningThreshold, numClasses ) )

    elif ( option == "forest15" ) :
        for i in range(15) :
            trees.append( top_DTL( trainingData, "randomized", pruningThreshold, numClasses ) )

    else :
        print("Option parameter not valid\nExiting...\n")
        return

    # Print output for learning phase
    for index, i in enumerate(trees) :
        printTree(i, index + 1)

    ### Test Phase ###
    testData, dummy = parceFile(testFile)

    accuracies = []
    for i in testData :
        prediction, accuracy = classifyObject(i, trees)
        accuracies.append(accuracy)
        print( "ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n" % ( testData.index(i), prediction, i[-1], accuracy ) )

    print( "classification accuracy=%6.4f\n" % ( sum(accuracies) / len(accuracies) ) )

    return

### Main ###
decision_tree( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] )