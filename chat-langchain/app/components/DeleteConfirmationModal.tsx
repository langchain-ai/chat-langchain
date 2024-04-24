import {
  Modal,
  Button,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
} from "@chakra-ui/react";

interface IDeleteConfirmationModal {
  isOpen: boolean;
  handleClose: () => void;
  handleDelete: () => void;
}

const DeleteConfirmationModal = ({
  isOpen,
  handleClose,
  handleDelete,
}: IDeleteConfirmationModal) => {
  return (
    <Modal isOpen={isOpen} onClose={handleClose}>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Delete Converation</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          Are you sure you want to delete the conversation? This action cannot
          be undone.
        </ModalBody>

        <ModalFooter>
          <Button colorScheme="red" mr={3} onClick={handleDelete}>
            Delete
          </Button>
          <Button variant="ghost" onClick={handleClose}>
            Cancel
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default DeleteConfirmationModal;
