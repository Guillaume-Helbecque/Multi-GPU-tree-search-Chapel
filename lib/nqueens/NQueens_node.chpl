/*******************************************************************************
Implementation of N-Queens Nodes.
*******************************************************************************/

module NQueens_node
{
  config param MAX_QUEENS = 20;

  record Node {
    var depth: uint(8);
    var board: MAX_QUEENS*uint(8);

    // default initializer
    proc init() {};

    // root initializer
    proc init(const N: int) {
      init this;
      for i in 0..#N do this.board[i] = i:uint(8);
    }

    /*
      NOTE: This copy-initializer makes the Node type "non-trivial" for `noinit`.
      Perform manual copy in the code instead.
    */
    // copy initializer
    /* proc init(other: Node) {
      this.depth = other.depth;
      this.board = other.board;
    } */
  }
}
