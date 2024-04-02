# deeplearning4j-2024
Deeplearning4j Workshop für den MATHEMA Campus 2024

In der Location werden zwar alle Zugang zum Internet haben, die Vergangenheit hat aber gezeigt, dass
* die herunterzuladende Datenmenge recht groß ist und
* das Netz dann überlastet wird, wenn alle Teilnehmenden gleichzeitig die Daten anfordern.

Deshalb empfehle ich, die oben genannten Schritte einen Tag vor dem Workshop zu Hause durchzuführen, damit die Übungen selbständig mitprogrammiert werden können.


# Setup
1. **IDE der Wahl installieren**
   1. im Workshop werde ich mit [IntelliJ IDEA](https://www.jetbrains.com/idea/download/) arbeiten 
   2. entweder die 30-Tage Testversion der Ultimate Edition oder die Community Edition herunterladen
2. **Buildtool installieren**
   1. optional, wenn IntelliJ verwendet wird
   2. ich werde [Maven](https://maven.apache.org/download.cgi) verwenden
   3. bitte die letzte stabile 3.x-Version verwenden
3. **DeepLearning4j-Beispiele klonen**
   1. [Deeplearning4j-Examples](https://github.com/deeplearning4j/deeplearning4j-examples.git)
4. **Abhängigkeiten auflösen und herunterladen**
   1. in das Verzeichnis `deeplearning4j-examples/dl4j-examples` wechseln und dort
   2. `mvn clean install` ausführen