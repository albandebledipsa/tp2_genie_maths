# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 14:07:06 2021

@author: alban Debled, Lara Chouraqui
"""

import numpy as np
from numpy import linalg as LA
import time
import matplotlib.pyplot as plt
from math import *

"""Partie 1 : Gauss"""

def ReductionGauss(Aaug):
    A = np.array(Aaug)
    for j in range(len(A[0])-1):
        for i in range(j+1, len(A)):
            pivot = A[j, j]
            g = A[i, j] / pivot
            A[i] = A[i] - g * A[j]
    return A

def ResolutionSystTriSup(Taug):
        
    n,m = np.shape(Taug)
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        somme = 0
        for j in range(i, n):
            somme += Taug[i][j] * X[j]
        X[i] = (Taug[i,n] - somme) / Taug[i][i]
    return X

def Gauss(matrice_augmente, A, B):
    startt_CPU = time.process_time()
    startt_eff = time.time()
    resultat = ReductionGauss(matrice_augmente)
    solution = ResolutionSystTriSup(resultat)
    stopt_CPU = time.process_time()
    stopt_eff = time.time()
    erreur = calcul_erreur(A, solution, B)
    temps = stopt_eff - startt_eff
    return temps, erreur

"""partie 2 : décomposition LU"""


def DecompositionLU(A):
    U = np.array(A)
    L = np.zeros((np.size(A[0]),np.size(A[0])))
    np.fill_diagonal(L, 1)    
    for j in range(len(U[0])):
        for i in range(j+1, len(U)):
            g = U[i, j] / U[j, j]
            U[i] = U[i] - g * U[j]
            L[i, j] = g
    return U,L


def ResolutionLU(L,U,B):

    n = len(B)
    X = np.zeros(n)
    Y = np.zeros(n)
    
    for i in range(0, n, 1):
        somme = 0
        for j in range(0 , i+1, 1):
            somme += L[i, j] * Y[j]
        Y[i] = (B[i] - somme) / L[i, i]

    for i in range(n-1, -1, -1):
        somme = 0
        for j in range(i , n):
            somme += U[i, j] * X[j]
        X[i] = (Y[i] - somme) / U[i, i]
    return X


def LU(A, B):
    startt_CPU = time.process_time()
    startt_eff = time.time()
    [U, L] = DecompositionLU(A)
    solution = ResolutionLU(L,U,B)
    stopt_CPU = time.process_time()
    stopt_eff = time.time()
    erreur = calcul_erreur(A, solution, B)
    temps = stopt_eff - startt_eff 
    return temps, erreur
    
""" Partie 3 : Gauss pivot partiel"""

def ReductionGaussPivotPartiel(Aaug):
    A = np.array(Aaug)
    m = 0
    for j in range(0, len(A)-1):
        L=[]
        for i in range(m, len(A)):
            L.append(A[i][j])
        pivot_max = L.index(max(L))
        if pivot_max != j:
            memoire = A[pivot_max + m, :].copy()
            A[pivot_max + m, :] = A[j, :]
            A[j, :] = memoire
        for i in range(j + 1, len(A)):
            pivot_max = A[j, j]
            g = A[i, j] / pivot_max
            A[i] = A[i] - g * A[j]
        m += 1
    return A

def ResolutionSystTriSupPivotPartiel(Taug):
    n,m = np.shape(Taug)
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        somme = 0
        for j in range(i, n):
            somme += Taug[i][j] * X[j]
        X[i] = (Taug[i,n] - somme) / Taug[i][i]
    return X

def GaussChoixPivotPartiel(matrice_augmente, A, B):
    startt_CPU = time.process_time()
    startt_eff = time.time()
    resultat = ReductionGaussPivotPartiel(matrice_augmente)
    solution = ResolutionSystTriSupPivotPartiel(resultat)
    stopt_CPU = time.process_time()
    stopt_eff = time.time()
    erreur = calcul_erreur(A, solution, B)
    temps = stopt_eff - startt_eff 
    return temps, erreur

    
""" Partie 4 : "Gauss pivot total"""

def ReductionGaussPivotTotal(Aaug):
    A = np.array(Aaug)
    m = 0 
    for j in range(0, len(A)-1):
        L = []
        C = []
        for i in range(m, len(A)):
            L.append(A[i][j])
        for k in range(m, len(A)):
            C.append(A[m][k])
        pivot_max_ligne = L.index(max(L))
        pivot_max_colonne = C.index(max(C))
        if L[pivot_max_ligne] > C[pivot_max_colonne] :
            memoire = A[pivot_max_ligne + m, :].copy()
            A[pivot_max_ligne + m, :] = A[j, :]
            A[j, :] = memoire
        if L[pivot_max_ligne] < C[pivot_max_colonne] :
            memoire = A[:, pivot_max_colonne + m].copy()
            A[:, pivot_max_colonne + m] = A[: ,j]
            A[:, j] = memoire
        if L[pivot_max_ligne] == C[pivot_max_colonne] :
            memoire = A[pivot_max_ligne + m, :].copy()
            A[pivot_max_ligne + m, :] = A[j, :]
            A[j, :] = memoire
        
        for i in range(j + 1, len(A)):
            pivot_max = A[j, j]
            g = A[i, j] / pivot_max
            A[i] = A[i] - g * A[j]
        m += 1
    return A

def ResolutionSystTriSupPivotTotal(Taug):
    
    n,m = np.shape(Taug)
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        somme = 0
        for j in range(i, n):
            somme += Taug[i][j] * X[j]
        X[i] = (Taug[i,n] - somme) / Taug[i][i]
    return X

def reorganisation_solution(A, resultat, solution_desordre):
    for k in range(len(A)):
        for l in range(len(A)):
            if resultat[0,0] == A[k, l]:
                indice = k
                break
    for i in range(len(A[0])):
        if A[indice, i] != resultat[0, i]:
            for j in range(i, len(A)):
                if A[indice, i] == resultat[0, j]:
                    resultat[0, i], resultat[0, j] = resultat[0, j], resultat[0, i]
                    solution_desordre[i], solution_desordre[j] = solution_desordre[j], solution_desordre[i]
                    break
    return solution_desordre
        
        
    

def GaussChoixPivotTotal(matrice_augmente, A, B):
        startt_CPU = time.process_time()
        startt_eff = time.time()
        resultat = ReductionGaussPivotTotal(matrice_augmente)
        solution_desordre = ResolutionSystTriSupPivotTotal(resultat)
        solution = reorganisation_solution(A, resultat, solution_desordre)
        stopt_CPU = time.process_time()
        stopt_eff = time.time()
        erreur = calcul_erreur(A, solution, B)
        temps = stopt_eff - startt_eff
        return temps, erreur


"""Partie 5 : linalg_solve"""

def linalg_solve(A, B):
        startt_CPU = time.process_time()
        startt_eff = time.time()
        X = LA.solve(A, B)
        stopt_CPU = time.process_time()
        stopt_eff = time.time()
        erreur = calcul_erreur(A, X, B)
        temps = stopt_eff - startt_eff
        return temps, erreur

"""Partie 6 : cholesky machine"""

def Cholesky_machine(A):
    L = np.linalg.cholesky(A)
    return(L)

def ResolCholesky_machine(A, B):
    n = len(B)
    X = np.zeros(n)
    Y = np.zeros(n)
    
    L = Cholesky_machine(A)
    Lt = np.transpose(L)
    
    for i in range(0, n, 1):
        somme = 0
        for j in range(0 , i+1, 1):
            somme += L[i, j] * Y[j]
        Y[i] = (B[i] - somme) / L[i, i]
        
    for i in range(n-1, -1, -1):
        somme = 0
        for j in range(i , n):
            somme += Lt[i, j] * X[j]
        X[i] = (Y[i] - somme) / Lt[i, i]
    return X

def main_cholesky_machine(A, B):

    startt_eff = time.time()
    solution = ResolCholesky_machine(A, B)
    stopt_eff = time.time()
    erreur = calcul_erreur(A, solution, B)
    temps = stopt_eff - startt_eff
    return temps, erreur
    
"""Partie 7 : cholesky"""

def Cholesky(A):
    L = np.zeros((np.size(A[0]),np.size(A[0])))
    for j in range(len(A)):
        for i in range(j, len(A)):
            somme = 0
            for k in range(j):
                somme += L[i,k]*L[j,k]
            if i == j:
                L[i,j] = sqrt(A[j,j] - somme)
            else:
                L[i,j] = (A[i,j] - somme)/L[j,j]
    return(L)
    
def ResolCholesky(A, B):
    n = len(B)
    X = np.zeros(n)
    Y = np.zeros(n)
    
    L = Cholesky(A)
    Lt = np.transpose(L)
    
    for i in range(0, n, 1):
        somme = 0
        for j in range(0 , i+1, 1):
            somme += L[i, j] * Y[j]
        Y[i] = (B[i] - somme) / L[i, i]
        
    for i in range(n-1, -1, -1):
        somme = 0
        for j in range(i , n):
            somme += Lt[i, j] * X[j]
        X[i] = (Y[i] - somme) / Lt[i, i]
    return X

    
def main_cholesky(A, B):
    startt_eff = time.time()
    solution = ResolCholesky(A, B)
    stopt_eff = time.time()
    erreur = calcul_erreur(A, solution, B)
    temps = stopt_eff - startt_eff
    return temps, erreur


""" Programme principal"""

def main():
    temps_Gauss = []
    temps_LU = []
    temps_pivot_partiel = []
    temps_pivot_total = []
    temps_linalg_solve = []
    temps_cholesky_machine = []
    temps_cholesky = []
    erreur_Gauss = []
    erreur_LU = []
    erreur_pivot_partiel = []
    erreur_pivot_total = []
    erreur_linalg_solve = []
    erreur_cholesky_machine = [] 
    erreur_cholesky = []
    taille_de_la_matrice = []
    
    
    for i in range(200, 1000, 50):
        
        print(i)
        M = []
        B = []
        A = []
        M = np.random.rand(i,i)
        Mt = np.transpose(M)
        A = np.dot(M, Mt)
        B = np.random.rand(i,1)
        matrice_augmente = np.append(A, B, axis = 1)
        
        taille_de_la_matrice.append(i)
        solution1 = Gauss(matrice_augmente, A, B)
        solution2 = LU(A, B)
        solution3 = GaussChoixPivotPartiel(matrice_augmente, A, B)
        solution4 = linalg_solve(A, B)
        solution5 = GaussChoixPivotTotal(matrice_augmente, A, B)
        solution6 = main_cholesky(A, B)
        solution7 = main_cholesky_machine(A, B)
        temps_Gauss.append(solution1[0])
        erreur_Gauss.append(solution1[1])
        temps_LU.append(solution2[0])
        erreur_LU.append(solution2[1])
        temps_pivot_partiel.append(solution3[0])
        erreur_pivot_partiel.append(solution3[1])
        temps_linalg_solve.append(solution4[0])
        erreur_linalg_solve.append(solution4[1])
        temps_pivot_total.append(solution5[0])
        erreur_pivot_total.append(solution5[1])
        temps_cholesky.append(solution6[0])
        erreur_cholesky.append(solution6[1])    
        temps_cholesky_machine.append(solution7[0])
        erreur_cholesky_machine.append(solution7[1])   
          
    fig, ax = plt.subplots()
    ax.plot(taille_de_la_matrice, temps_Gauss, label = "Gauss")
    ax.plot(taille_de_la_matrice, temps_LU, label = "LU")
    ax.plot(taille_de_la_matrice, temps_pivot_partiel, label = "Gauss pivot partiel")
    ax.plot(taille_de_la_matrice, temps_pivot_total, label = "Gauss pivot total")    
    ax.plot(taille_de_la_matrice, temps_linalg_solve, label = "linalg_solve")
    ax.plot(taille_de_la_matrice, temps_cholesky, label = "Cholesky")
    ax.plot(taille_de_la_matrice, temps_cholesky_machine, label = "Cholesky machine")
        
    plt.legend()
    plt.title("Temps en fonction de la taille de la matrice pour toutes les méthodes")
    plt.xlabel("taille des matrices")
    plt.ylabel("temps (secondes)")
    plt.savefig("graph temps toutes méthodes")
    
    fig, ax2 = plt.subplots()
    ax2.plot(taille_de_la_matrice, erreur_Gauss, label = "Gauss")
    ax2.plot(taille_de_la_matrice, erreur_LU, label = "LU")
    ax2.plot(taille_de_la_matrice, erreur_pivot_partiel, label = "Gauss pivot partiel")
    ax2.plot(taille_de_la_matrice, erreur_pivot_total, label = "Gauss pivot total")    
    ax2.plot(taille_de_la_matrice, erreur_linalg_solve, label = "linalg_solve")
    ax2.plot(taille_de_la_matrice, erreur_cholesky, label = "Cholesky")
    ax2.plot(taille_de_la_matrice, erreur_cholesky_machine, label = "Cholesky machine")

    plt.legend()
    plt.title("Erreur en fonction de la taille de la matrice pour toutes les méthodes")
    plt.xlabel("taille des matrices")
    plt.ylabel("erreur")
    plt.savefig("graph erreur toutes méthodes")

    plt.show()
        

def calcul_erreur(A, X, B):
    produit = np.dot(A, X)
    for i in range(len(produit)):
        produit[i] = produit[i] - B[i]
    erreur = LA.norm(produit)
    return erreur

main()