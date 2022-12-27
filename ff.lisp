;; feed forward networks based on the book
;; Rashid (2016) Make your own neural network, CreateSpace Independent Publishing Platform.

;; our first representation of a network can be a sequence of 2d arrays
;; so we need to be able to make an array of any two-dimensions and then
;; initialize it with random entries

;; matrix multiplication
;; to-do: throw an error if the dimensions don't match, but cs2 and rs1 should be equal
(defun matmul (a2 a1)
  (destructuring-bind (rs2 cs2) (array-dimensions a2)
    (destructuring-bind (rs1 cs1) (array-dimensions a1)
      (let ((new-array (make-array (list rs2 cs1))))
        (dotimes (r2 rs2)
          (dotimes (c1 cs1)
            (let ((sum 0))
              (dotimes (r1 rs1)
                (incf sum (* (aref a2 r2 r1) (aref a1 r1 c1))))
              (setf (aref new-array r2 c1) sum))))
        new-array))))



(defun transpose (a)
  (destructuring-bind (rs cs) (array-dimensions a)
    (let ((new-array (make-array (list cs rs))))
      (dotimes (r rs)
        (dotimes (c cs)
          (setf (aref new-array c r) (aref a r c))))
      new-array)))

(defun zeros (inodes onodes)
  (make-array (list onodes inodes) :initial-element 0))

;; specifically 2d version
(defun map-array-2d (fn a)
  (destructuring-bind (rs cs) (array-dimensions a)
    (let ((new-array (make-array (list rs cs) :initial-element 0)))
      (dotimes (r rs)
        (dotimes (c cs)
          (setf (aref new-array r c) (funcall fn (aref a r c)))))
      new-array)))
        
;; this is the general version that uses the total array size and row major order
(defun map-array (fn a)
  (let ((s (array-total-size a))
        (new-array (make-array (array-dimensions a))))
    (dotimes (i s)
      (setf (row-major-aref new-array i) (funcall fn (row-major-aref a i))))
    new-array))

(defun random-array (inodes onodes)
  (map-array #'(lambda (x) (random 1.0)) (zeros inodes onodes)))

;; we're going to use the inodes and onodes description in order to try and make it clear how all of this works

(defun zip (l1 l2)
  (mapcar #'(lambda (x y) (list x y)) l1 l2))

(defun connection-sizes (ls)
  (zip ls (cdr ls)))

(defun make-ff-net (layers)
  (mapcar #'(lambda (p) (apply #'random-array p)) (connection-sizes layers)))
